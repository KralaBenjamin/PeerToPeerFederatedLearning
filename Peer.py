import sys
import asyncio
import pickle
import logging
import random
from collections import deque

logging.basicConfig(level=logging.INFO)


class PeerNode:
    def __init__(self, host, port, max_peers=2, max_connections=2, max_classes=10):
        self.host = host
        self.port = port
        self.max_peers = max_peers              # number of maximal incoming connections that a peer accepts

        self.max_connections = max_connections  # number of maximal outgoing connections that a peer tries to establish

        # Keeps track of peers that have an incoming active connection to this peer
        # These peers can request weights
        # Entries of form (host, port):[classes]
        self.peers = {}

        # Keep track of the connected peers (outgoing connections)
        # This peer can request weights from the connected peers
        # Entries of form (host, port):(reader, writer) with reader and writer objects from asyncio-connection
        self.connections = {}

        self.classes = []       # classes (ML) of which data is present at the node
        # Keep track of classes connected peers have present
        # TODO make more general, instantiable for not only numbers but arbitrary class names?
        self.connected_classes = {key: 0 for key in range(max_classes)}

        # Keep track of failed connection attempts to avoid multiple tries
        # instantiated with own listening address to avoid connection attempts of peer to itself
        self.connections_refused = [(self.host, self.port)]

        # Keep track of recent incoming connection attempts to give as new peers
        # Queue of fixed length: always keep most current entries
        self.current_connections = deque(maxlen=10*max_connections)

        self.weights = None     # weights (ML) to share when requested

    async def start_server(self):
        server = await asyncio.start_server(self.handle_connection, self.host, self.port)
        logging.info(f"Node listening on {self.host}:{self.port}")

        async with server:
            await server.serve_forever()

    async def receive_package(self, reader):
        package_length_data = await reader.readexactly(4)
        package_length = int.from_bytes(package_length_data, "big")

        package_data = await reader.readexactly(package_length)
        package = pickle.loads(package_data)

        address = package["ADDRESS"]
        action = package["ACTION"]
        load = package["LOAD"]

        logging.info(f"Received message from {address} - Action: {action}, Load: {load}")

        return package

    async def send_package(self, writer, action="", load=None):
        package = {
            "ADDRESS": (self.host, self.port, self.classes),
            "ACTION": action,
            "LOAD": load
        }
        package_data = pickle.dumps(package)
        package_length = len(package_data).to_bytes(4, "big")

        writer.write(package_length)
        writer.write(package_data)
        await writer.drain()

    async def handle_request(self, package, writer):
        if package["ACTION"] == "REQUEST WEIGHTS":
            if self.weights is not None:
                await self.send_package(writer, "WEIGHTS SEND", self.weights)

            else:
                await self.send_package(writer, "UNABLE TO SEND WEIGHTS")
        elif package["ACTION"] == "SEEK PEERS":
            await self.send_package(writer, "PEERS SEND", list(self.current_connections))

    async def handle_response(self, package):
        if package["ACTION"] == "WEIGHTS SEND":
            weights = package["LOAD"]
            # TODO do something with received weights
            # TODO what do if peer was unable to send weights?
            # Either request different connection of retry after x seconds with asyncio.sleep()

    # Method is called automatically when a connection the listening port (server) is established
    # Initial incoming messages can be: "SEEK PEERS", "SEEK CONNECTION"
    async def handle_connection(self, reader, writer):
        peer_host, peer_port = writer.get_extra_info('peername')
        logging.info(f"Incoming connection from {peer_host}:{peer_port}")

        initial_package = await self.receive_package(reader)

        self.current_connections.append(initial_package["ADDRESS"])

        if len(self.peers) < self.max_peers:
            # TODO Enum?
            # Return "PEERS SEND" with list self.current_connections
            # including this node, because it still has space for connections
            if initial_package["ACTION"] == "SEEK PEERS":
                package_load = list(self.current_connections.copy())
                package_load.append((self.host, self.port, self.classes))
                await self.send_package(writer, "PEERS SEND", package_load)
                logging.info(f"Send peer list to {peer_host}:{peer_port}")
                writer.close()
                await writer.wait_closed()

            # Still space for connection: return "CONNECTION ACCEPTED"
            elif initial_package["ACTION"] == "SEEK CONNECTION":
                await self.send_package(writer, "CONNECTION ACCEPTED", None)
                logging.info(f"Accepting connection from {peer_host}:{peer_port}")

                # Update list self.peers
                host, port, classes = initial_package["ADDRESS"]
                self.peers[(host, port)] = classes

                # Established connection: wait for receiving packages (requests) in infinite loop
                try:
                    while True:
                        package = await self.receive_package(reader)
                        await self.handle_request(package, writer)

                except asyncio.IncompleteReadError as e:
                    logging.error(f"Incomplete Read error: {e}")
                    pass
        # No more connections possible: send list self.current_connections and close connection
        # Depending on incoming message, either respond with "PEERS SEND" or "CONNECTION REFUSED"
        else:
            logging.info(f"Max peers limit reached for {peer_host}:{peer_port}")
            await self.send_package(writer,
                                    "PEERS SEND" if initial_package["ACTION"] == "SEEK PEERS" else "CONNECTION REFUSED",
                                    list(self.current_connections))
            writer.close()
            await writer.wait_closed()

        logging.info(f"Connection with {peer_host}:{peer_port} closed")
        writer.close()
        await writer.wait_closed()

    # Method for trying to establish a connection to a peer
    # Only for initially entering the network (bootstrapping), init should be set True
    async def connect_to_peer(self, peer_host, peer_port, init=False):
        try:
            reader, writer = await asyncio.open_connection(peer_host, peer_port)
            logging.info(f"Connected to peer {peer_host}:{peer_port}")

            # For bootstrapping, send "SEEK PEERS". Else, send "SEEK CONNECTION"
            await self.send_package(writer, "SEEK PEERS" if init else "SEEK CONNECTION")

            response_package = await self.receive_package(reader)
            package_load = response_package["LOAD"]

            # List of peers from bootstrapping peer was received
            # Choose self.max_connections peers from list randomly
            if response_package["ACTION"] == "PEERS SEND":
                # TODO what if len(package_load) < number required connections
                #  now: ignore this, in future: second SEEK PEERS to other peer?
                for i in range(min(self.max_connections-len(self.connections), len(package_load))):
                    if package_load:
                        random_peer = random.choice(package_load)           # select random peer
                        package_load.remove(random_peer)                    # remove selected from list
                        random_peer_host, random_peer_port, _ = random_peer
                        # Check if this connection was tried before already, if so: jump to next random peer
                        if (random_peer_host, random_peer_port) in self.connections.keys() \
                                or (random_peer_host, random_peer_port) in self.connections_refused:
                            continue
                        # Here, the actual connections are requested
                        asyncio.create_task(self.connect_to_peer(random_peer_host, random_peer_port))

            # Connection to specific peer could not be established: select one peer from send list
            elif response_package["ACTION"] == "CONNECTION REFUSED":
                self.connections_refused.append((peer_host, peer_port))
                writer.close()
                await writer.wait_closed()

                if package_load:
                    peers_sorted = self.get_classes_order(package_load)
                    for (host, port) in peers_sorted:
                        if (host, port) in self.connections.keys() \
                                or (host, port) in self.connections_refused:
                            continue

                        asyncio.create_task(self.connect_to_peer(host, port))
                        break

            # Connection to peer was successful
            elif response_package["ACTION"] == "CONNECTION ACCEPTED":
                # update self.connections and self.connected_classes
                self.connections[(peer_host, peer_port)] = (reader, writer)
                _, _, classes = response_package["ADDRESS"]
                for c in classes:
                    self.connected_classes[c] += 1

                # Wait for responses from connected peer in infinite loop
                while True:
                    package = await self.receive_package(reader)
                    await self.handle_response(package)

        except (ConnectionRefusedError, asyncio.TimeoutError):
            logging.error(f"Failed to connect to peer {peer_host}:{peer_port}")

    # sort dict of (host, port):[classes] by ranking classes
    # depending on self.classes and frequency in self.connected_classes
    def get_classes_order(self, package_load):
        classes_dict = {}
        for host, port, classes in package_load:
            peer_rank = 0.0
            for c in classes:
                # for each class that is not present as data at peer: +1
                if c not in self.classes:
                    peer_rank += 1.0
                # for classes that already established connections have: rank descending by amount
                peer_rank += 1.0/(self.connected_classes[c]+1.0)
            classes_dict[(host, port)] = peer_rank

        return sorted(classes_dict, reverse=True)


async def main(node_port, bootstrap_port): # DOKU: Startet neuen Peer
    node = PeerNode('localhost', node_port, max_peers=2)

    server_task = asyncio.create_task(node.start_server())

    # Wait for the servers to start (optional)
    await asyncio.sleep(1)

    # Connect to other peers
    if bootstrap_port is not None:
        asyncio.create_task(node.connect_to_peer('localhost', bootstrap_port, True))

    # Wait for connections to establish
    await asyncio.sleep(10)

    # TODO get rid of while True-loop
    # TODO implement better strategy instead of random querying
    # Start sending/receiving messages with connected peers
    # Wait random time, Select random connected peer and ask for weights

    if len(node.connections) > 0:
        while True:
            await asyncio.sleep(random.randint(5, 10))
            connection_id, (reader, writer) = random.choice(list(node.connections.items()))
            await node.send_package(writer, "REQUEST WEIGHTS")

    await server_task


if __name__ == '__main__':
    port = 8000
    connecting_port = None

    if len(sys.argv) >= 2:
        port = sys.argv[1]
    if len(sys.argv) >= 3:
        connecting_port = sys.argv[2]

    asyncio.run(main(port, connecting_port))
