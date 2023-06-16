import sys
import asyncio
import pickle
import logging

logging.basicConfig(level=logging.INFO)


class PeerNode:
    def __init__(self, host, port, max_peers=2):
        self.host = host
        self.port = port
        self.max_peers = max_peers
        self.peers = set()
        self.connections = {}

    async def start(self):
        server = await asyncio.start_server(self.handle_connection, self.host, self.port)
        logging.info(f"Node listening on {self.host}:{self.port}")

        async with server:
            await server.serve_forever()

    async def receive_package(self, reader, peer_host, peer_port):
        package_length_data = await reader.readexactly(4)
        package_length = int.from_bytes(package_length_data, "big")

        package_data = await reader.readexactly(package_length)
        package = pickle.loads(package_data)

        action = package["ACTION"]
        load = package["LOAD"]

        logging.info(f"Received message from {peer_host}:{peer_port} - Action: {action}, Load: {load}")

        return package

    async def send_package(self, writer, action="", load=None):
        package = {
            "ADDRESS": (self.host, self.port),
            "ACTION": action,
            "LOAD": load
        }
        package_data = pickle.dumps(package)
        package_length = len(package_data).to_bytes(4, "big")

        writer.write(package_length)
        writer.write(package_data)
        await writer.drain()

    async def handle_connection(self, reader, writer):
        peer_host, peer_port = writer.get_extra_info('peername')
        logging.info(f"Incoming connection from {peer_host}:{peer_port}")

        initial_package = await self.receive_package(reader, peer_host, peer_port)

        if len(self.peers) < self.max_peers:
            logging.info(f"Accepting connection from {peer_host}:{peer_port}")

            response_load = None
            if initial_package["ACTION"] == "SEEK PEERS":
                response_load = ",".join([f"{host}:{port}" for host, port in self.peers])

            await self.send_package(writer, "CONNECTION ACCEPTED", response_load)

            self.peers.add(initial_package["ADDRESS"])

            try:
                while True:
                    package = await self.receive_package(reader, peer_host, peer_port)

                    action = package["ACTION"]
                    load = package["LOAD"]

                    logging.info(f"Received message from {peer_host}:{peer_port} - Action: {action}, Load: {load}")

                    # Handle the received message here

            except asyncio.IncompleteReadError as e:
                logging.error(f"Incomplete Read error: {e}")
                pass

        else:
            logging.error(f"Max peers limit reached for {peer_host}:{peer_port}")
            response_load = ",".join([f"{host}:{port}" for host, port in self.peers])

            await self.send_package(writer, "CONNECTION REFUSED", response_load)

        logging.info(f"Connection with {peer_host}:{peer_port} closed")
        writer.close()
        await writer.wait_closed()

    async def connect_to_peer(self, peer_host, peer_port, init=False):
        try:
            reader, writer = await asyncio.open_connection(peer_host, peer_port)

            logging.info(f"Connected to peer {peer_host}:{peer_port}")

            await self.send_package(writer, "SEEK PEERS" if init else "SEEK CONNECTION")

            connection_id = f"{peer_host}:{peer_port}"
            self.connections[connection_id] = (reader, writer)

            response_package = await self.receive_package(reader, peer_host, peer_port)

            if response_package["ACTION"] == "CONNECTION ACCEPTED":
                connection_id = f"{peer_host}:{peer_port}"
                self.connections[connection_id] = (reader, writer)

            received_load = response_package["LOAD"]
            if received_load:
                received_peers = received_load.split(",")
                for received_peer in received_peers:
                    if received_peer:
                        host, peer = received_peer.split(":")
                        asyncio.create_task(self.connect_to_peer(host, peer))

            if response_package["ACTION"] == "CONNECTION ACCEPTED":
                connection_id = f"{peer_host}:{peer_port}"
                self.connections[connection_id] = (reader, writer)
                while True:
                    package = await self.receive_package(reader, peer_host, peer_port)
                # Start sending/receiving messages with the connected peer

        except (ConnectionRefusedError, asyncio.TimeoutError):
            logging.error(f"Failed to connect to peer {peer_host}:{peer_port}")

        finally:
            writer.close()
            await writer.wait_closed()

    def send_message(self, connection_id, action, load=None):
        if connection_id in self.connections:
            _, writer = self.connections[connection_id]

            package = {
                "ADDRESS": (self.host, self.port),
                "ACTION": action,
                "LOAD": load
            }
            package_data = pickle.dumps(package)
            package_length = len(package_data).to_bytes(4, "big")

            writer.write(package_length)
            writer.write(package_data)
            return True

        return False


async def main(port, connecting_port):

    node = PeerNode('localhost', port, max_peers=2)

    server_task = asyncio.create_task(node.start())

    # Wait for the servers to start (optional)
    await asyncio.sleep(1)

    # Connect to other peers
    if connecting_port is not None:
        asyncio.create_task(node.connect_to_peer('localhost', connecting_port, True))

    # Start sending/receiving messages with connected peers

    await server_task

if __name__ == '__main__':
    port = 8000
    connecting_port = None

    if len(sys.argv) >= 2:
        port = sys.argv[1]
    if len(sys.argv) >= 3:
        connecting_port = sys.argv[2]

    asyncio.run(main(port, connecting_port))