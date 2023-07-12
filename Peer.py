import sys
import asyncio
import pickle
import logging
import random
from collections import deque
from ml_class import MLModell

logging.basicConfig(level=logging.INFO)


class PeerNode:
    def __init__(self, host, port, max_peers=2,
                 max_connections=2, num_classes=3, max_classes=10,
                 num_epochs=2, ml_type='avg',
                 num_training_samples=128, num_test_samples=512):
        self.model = None
        self.bootstrap_port = None
        self.host = host
        self.port = port
        self.max_peers = max_peers  # number of maximal incoming connections that a peer accepts

        self.max_connections = max_connections  # number of maximal outgoing connections that a peer tries to establish

        # ML settings
        self.num_epochs = num_epochs
        self.num_training_samples = num_training_samples
        self.num_test_samples = num_test_samples
        self.ml_type = ml_type
        self.num_classes = num_classes

        # Keeps track of peers that have an incoming active connection to this peer
        # These peers can request weights
        # Entries of form (host, port):[classes]
        self.peers = {}

        # Keep track of the connected peers (outgoing connections)
        # This peer can request weights from the connected peers
        # Entries of form (host, port):(reader, writer) with
        # reader and writer objects from asyncio-connection and requested weights
        self.connections = {}

        # Keep track of the received state dicts
        # Entries of form (host, port):(received_weights) with
        # reader and writer objects from asyncio-connection and requested weights
        self.received_state_dicts = {}

        self.classes = []  # classes (ML) of which data is present at the node
        # Keep track of classes connected peers have present
        # TODO make more general, instantiable for not only numbers but arbitrary class names?
        self.connected_classes = {key: 0 for key in range(max_classes)}

        # Keep track of failed connection attempts to avoid multiple tries
        # instantiated with own listening address to avoid connection attempts of peer to itself
        self.connections_refused = [(self.host, self.port)]

        # Keep track of recent incoming peer connection attempts to use as new peers
        # Queue of fixed length: always keep most current entries
        self.current_peers = deque(maxlen=10 * max_connections)

    def initialize_ml_stuff(self):
        self.model = MLModell(num_epochs=self.num_epochs,
                              num_train_samples=self.num_training_samples,
                              num_test_samples=self.num_test_samples,
                              num_classes=self.num_classes)
        self.classes = self.model.classes

        print('Initializing some ml...')
        self.model.train()

    async def start_server(self):
        # start server on listening port

        self.initialize_ml_stuff()

        server = await asyncio.start_server(self.handle_connection, self.host, self.port)
        logging.info(f"Node listening on {self.host}:{self.port}")

        # create ml query avergae loop
        asyncio.create_task(self.query_combine_loop())

        # start soft state checking here
        asyncio.create_task(self.check_connections_soft_state())

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

        logging.info(f"Received message from {address} - Action: {action},"
                     f"Load: {load if action != 'WEIGHTS SEND' else '<STATE_DICT>'}")

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
        if writer.transport._conn_lost:
            writer.close()
            await writer.wait_closed()
        else:
            await writer.drain()
        del package, package_data

    async def handle_request(self, package, writer):
        if package["ACTION"] == "REQUEST WEIGHTS":
            if self.model is not None:
                await self.send_package(writer, "WEIGHTS SEND", self.model.get_current_weights())
            else:
                await self.send_package(writer, "UNABLE TO SEND WEIGHTS")
        elif package["ACTION"] == "SEEK PEERS":
            await self.send_package(writer, "PEERS SEND", list(self.current_peers))

    async def handle_response(self, package, connection_host, connection_port):
        if package["ACTION"] == "WEIGHTS SEND":
            self.received_state_dicts[(connection_host, connection_port)] = package["LOAD"]
        elif package["ACTION"] == "UNABLE TO SEND WEIGHTS":
            self.received_state_dicts[(connection_host, connection_port)] = None

    # Method is called automatically when a connection the listening port (server) is established
    # Initial incoming messages can be: "SEEK PEERS", "SEEK CONNECTION"
    async def handle_connection(self, reader, writer):
        peer_info = writer.get_extra_info('peername')
        logging.info(f"Incoming connection from {peer_info}")

        initial_package = await self.receive_package(reader)

        if initial_package["ADDRESS"] not in self.current_peers:
            self.current_peers.append(initial_package["ADDRESS"])

        if len(self.peers) < self.max_peers:
            # Return "PEERS SEND" with list self.current_connections
            # including this node, because it still has space for connections
            if initial_package["ACTION"] == "SEEK PEERS":
                package_load = list(self.current_peers.copy())
                package_load.append((self.host, self.port, self.classes))
                await self.send_package(writer, "PEERS SEND", package_load)
                logging.info(f"Send peer list to {peer_info}")
                writer.close()
                await writer.wait_closed()

            # Still space for connection: return "CONNECTION ACCEPTED"
            elif initial_package["ACTION"] == "SEEK CONNECTION":
                await self.send_package(writer, "CONNECTION ACCEPTED", None)
                logging.info(f"Accepting connection from {peer_info}")

                # Update list self.peers
                host, port, classes = initial_package["ADDRESS"]
                self.peers[(host, port)] = classes

                # Established connection: wait for receiving packages (requests) in infinite loop
                try:
                    while True:
                        package = await self.receive_package(reader)
                        await self.handle_request(package, writer)

                except asyncio.IncompleteReadError as e:
                    logging.error(f"Incomplete Read error: {e}. Lost peer at {host, port}")
                    self.peers.pop((host, port), "")
                    pass
        # No more connections possible: send list self.current_connections and close connection
        # Depending on incoming message, either respond with "PEERS SEND" or "CONNECTION REFUSED"
        else:
            logging.info(f"Max peers limit reached for {peer_info}")
            await self.send_package(writer,
                                    "PEERS SEND" if initial_package["ACTION"] == "SEEK PEERS" else "CONNECTION REFUSED",
                                    list(self.current_peers))
            writer.close()
            await writer.wait_closed()

        logging.info(f"Connection with {peer_info} closed")
        writer.close()
        await writer.wait_closed()

    # Method for trying to establish a connection to a peer
    # Only for initially entering the network (bootstrapping), init should be set True
    async def connect_to_peer(self, peer_host, peer_port, init=False):
        if init and len(self.current_peers) == 0:
            self.current_peers.append((peer_host, peer_port, []))
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
                for i in range(min(self.max_connections - len(self.connections), len(package_load))):
                    if package_load:
                        random_peer = random.choice(package_load)  # select random peer
                        package_load.remove(random_peer)  # remove selected from list
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
                self.received_state_dicts[(peer_host, peer_port)] = None
                _, _, classes = response_package["ADDRESS"]
                for c in classes:
                    self.connected_classes[c] += 1

                # Wait for responses from connected peer in infinite loop
                try:
                    while True:
                        package = await self.receive_package(reader)
                        await self.handle_response(package, peer_host, peer_port)
                except asyncio.IncompleteReadError as e:
                    logging.error(f"IncompleteReadError: {e}. Lost connection to {peer_host, peer_port}")
                    self.connections.pop((peer_host, peer_port), "")

                    # select randomly new peer and initiate getting new connections
                    if len(self.current_peers) > 0:
                        (new_host, new_port, _) = random.choice(self.current_peers)
                        asyncio.create_task(self.connect_to_peer(new_host, new_port, init=True))
                    elif len(self.connections) > 0:
                        (new_host, new_port) = random.choice(list(self.connections.keys()))
                        asyncio.create_task(self.connect_to_peer(new_host, new_port, init=True))
                    pass

        except (ConnectionRefusedError, asyncio.TimeoutError):
            logging.error(f"Failed to connect to peer {peer_host}:{peer_port}")

    async def check_connections_soft_state(self):
        while True:
            await asyncio.sleep(10)
            logging.info(
                f"Soft State checked. {len(self.connections)} out of {self.max_connections} connections present.")

            if len(self.connections) < self.max_connections/2:
                # select randomly new peer and initiate getting new connections
                if len(self.current_peers) > 0:
                    (new_host, new_port, _) = random.choice(self.current_peers)
                    asyncio.create_task(self.connect_to_peer(new_host, new_port, init=True))
                elif len(self.connections) > 0:
                    (new_host, new_port) = random.choice(list(self.connections.keys()))
                    asyncio.create_task(self.connect_to_peer(new_host, new_port, init=True))
            elif len(self.connections) > self.max_connections:
                for i in range(len(self.connections)-self.max_connections):
                    (_, _), (reader, writer) = self.connections.popitem()
                    writer.close()
                    await writer.wait_closed()

    async def query_combine_loop(self):
        while True:
            await asyncio.sleep(0.1)
            if len(self.connections) > 0:
                for connected_node, (_, writer) in self.connections.items():
                    await self.send_package(writer, "REQUEST WEIGHTS")
                await asyncio.sleep(3)
                collected_state_dicts = []
                for connected_node, state_dict in self.received_state_dicts.items():
                    if state_dict is not None:
                        collected_state_dicts.append(state_dict)
                        self.received_state_dicts[connected_node] = None

                if len(collected_state_dicts) > 0:
                    logging.info(f"Averaging over {len(collected_state_dicts)} state dicts.")
                    if self.ml_type == "avg":
                        self.model.average(collected_state_dicts)
                    elif self.ml_type == "max":
                        self.model.select_max(collected_state_dicts)
                else:
                    logging.info(f"No state dicts received.")

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
                peer_rank += 1.0 / (self.connected_classes[c] + 1.0)
            classes_dict[(host, port)] = peer_rank

        return sorted(classes_dict, reverse=True)


async def main(node_port, bootstrap_port):  # DOKU: Startet neuen Peer

    node = PeerNode('localhost', node_port, max_peers=2)

    server_task = asyncio.create_task(node.start_server())

    # Wait for the servers to start (optional)
    await asyncio.sleep(1)

    # Connect to other peers
    if bootstrap_port is not None:
        asyncio.create_task(node.connect_to_peer('localhost', bootstrap_port, True))

    # Wait for connections to establish
    await asyncio.sleep(5)

    # Start sending/receiving messages with connected peers
    # Wait random time, query connected peer for weights, collect weights & average

    if len(node.connections) > 0:
        while True:
            await asyncio.sleep(random.randint(1, 5))
            for connected_node, (_, writer) in node.connections.items():
                await node.send_package(writer, "REQUEST WEIGHTS")
            await asyncio.sleep(3)
            collected_state_dicts = []
            for connected_node, state_dict in node.received_state_dicts.items():
                if state_dict is not None:
                    collected_state_dicts.append(state_dict)
                    node.received_state_dicts[connected_node] = None

            logging.info(f"Averaging over {len(collected_state_dicts)} state dicts.")
            node.model.average(collected_state_dicts)

    await server_task


if __name__ == '__main__':
    port = 8000
    connecting_port = None

    if len(sys.argv) >= 2:
        port = sys.argv[1]
    if len(sys.argv) >= 3:
        connecting_port = sys.argv[2]

    asyncio.run(main(port, connecting_port))
