import sys
import asyncio
import pickle
import logging
import random

logging.basicConfig(level=logging.INFO)


class PeerNode:
    def __init__(self, host, port, max_peers=2, max_connections=2, max_classes=10):
        self.host = host
        self.port = port
        self.max_peers = max_peers
        self.max_connections = max_connections
        self.peers = {} # Doku: Wissen über andere Peers
        self.connections = {} # Doku: Aktive Peers, mit denen wir verbunden haben
        self.classes = [1,2] # Doku: Klassen, die der Peer selber hat
        # TODO: Parameter für allgemeine Klassen geben.
        self.connected_classes = {key: 0 for key in range(max_classes+1)}
        self.connections_refused = [(self.host, self.port)]
        self.weights = None

    async def start(self):
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
            await self.send_package(writer, "PEERS SEND", self.peers)

    async def handle_response(self, package):
        if package["ACTION"] == "WEIGHTS SEND":
            weights = package["LOAD"]
            # TODO do something with received weights
            # TODO what do if peer was unable to send weights?
            # Either request different connection of retry after x seconds with asyncio.sleep()

    async def handle_connection(self, reader, writer):
        peer_host, peer_port = writer.get_extra_info('peername')
        logging.info(f"Incoming connection from {peer_host}:{peer_port}")

        initial_package = await self.receive_package(reader)

        if len(self.peers) <= self.max_peers:

            if initial_package["ACTION"] == "SEEK PEERS": #TODO: Dokumentieren, was die Strings bedeuten. Enum?
                package_load = self.peers.copy()
                package_load[(self.host, self.port)] = self.classes
                await self.send_package(writer, "PEERS SEND", package_load)
                logging.info(f"Send peer list to {peer_host}:{peer_port}")
                writer.close()
                await writer.wait_closed()

            elif initial_package["ACTION"] == "SEEK CONNECTION":
                await self.send_package(writer, "CONNECTION ACCEPTED", None)
                logging.info(f"Accepting connection from {peer_host}:{peer_port}")

                host, port, classes = initial_package["ADDRESS"]
                self.peers[(host, port)] = classes

                try:
                    while True:
                        package = await self.receive_package(reader)
                        await self.handle_request(package, writer)

                except asyncio.IncompleteReadError as e:
                    logging.error(f"Incomplete Read error: {e}")
                    pass

        else:
            logging.info(f"Max peers limit reached for {peer_host}:{peer_port}")
            await self.send_package(writer,
                                    "PEERS SEND" if initial_package["ACTION"] == "SEEK PEERS" else "CONNECTION REFUSED",
                                    self.peers)
            writer.close()
            await writer.wait_closed()

        logging.info(f"Connection with {peer_host}:{peer_port} closed")
        writer.close()
        await writer.wait_closed()

    async def connect_to_peer(self, peer_host, peer_port, init=False):
        try:
            reader, writer = await asyncio.open_connection(peer_host, peer_port)
            logging.info(f"Connected to peer {peer_host}:{peer_port}")

            await self.send_package(writer, "SEEK PEERS" if init else "SEEK CONNECTION")

            response_package = await self.receive_package(reader)
            package_load = response_package["LOAD"]

            # Choose maxpeers peers from send list based on their classes
            if response_package["ACTION"] == "PEERS SEND":
                if package_load:
                    peers_sorted = self.get_classes_order(package_load)
                    peers_contacted_counter = 0
                    for (host, port) in peers_sorted:
                        if peers_contacted_counter >= self.max_connections:
                            break

                        if (host, port) not in self.connections.keys() \
                                and (host, port) not in self.connections_refused:
                            peers_contacted_counter += 1
                            asyncio.create_task(self.connect_to_peer(host, port)) # DOKU: Hier werden eigentliche Verbindungen aufgebaut!

            # Connection to specific peer could not be established: select one peer from send list
            elif response_package["ACTION"] == "CONNECTION REFUSED":
                self.connections_refused.append((peer_host, peer_port))
                writer.close()
                await writer.wait_closed()

                if package_load:
                    peers_sorted = self.get_classes_order(package_load)
                    peers_contacted_counter = 0
                    for (host, port) in peers_sorted:
                        if peers_contacted_counter >= 1:
                            break

                        if (host, port) not in self.connections.keys() \
                                and (host, port) not in self.connections_refused:
                            peers_contacted_counter += 1
                            asyncio.create_task(self.connect_to_peer(host, port))
            # Connection to peer was successful
            elif response_package["ACTION"] == "CONNECTION ACCEPTED":
                self.connections[(peer_host, peer_port)] = (reader, writer)
                _, _, classes = response_package["ADDRESS"]
                for c in classes:
                    self.connected_classes[c] += 1

                while True:
                    package = await self.receive_package(reader)
                    await self.handle_response(package)
                # Start sending/receiving messages with the connected peer

        except (ConnectionRefusedError, asyncio.TimeoutError):
            logging.error(f"Failed to connect to peer {peer_host}:{peer_port}")

        finally:
            writer.close()
            await writer.wait_closed()

    def get_classes_order(self, package_load):
        classes_dict = {}
        for (host, port), classes in package_load.items():
            peer_rank = 0.0
            for c in classes:
                # für alle Klassen, die nicht haben, +1
                if c not in self.classes:
                    peer_rank += 1.0
                # für alle Klassen, die unsere Nachbarn haben
                # ranken wir. 
                peer_rank += 1.0/(self.connected_classes[c]+1.0)
            classes_dict[(host, port)] = peer_rank

        return sorted(classes_dict, reverse=True)


async def main(node_port, bootstrap_port): # DOKU: Startet neuen Peer
    node = PeerNode('localhost', node_port, max_peers=2)

    server_task = asyncio.create_task(node.start())

    # Wait for the servers to start (optional)
    await asyncio.sleep(1)

    # Connect to other peers
    if bootstrap_port is not None:
        asyncio.create_task(node.connect_to_peer('localhost', bootstrap_port, True))

    # Wait for connections to establish
    await asyncio.sleep(10)

    # TODO implement better strategy instead of random querying
    # Start sending/receiving messages with connected peers
    # Wait random time, Select random connected peer and ask for weights
    while True:
        if len(node.connections) > 0:
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
