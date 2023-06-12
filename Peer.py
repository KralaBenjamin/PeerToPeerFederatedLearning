import asyncio
import sys


class PeerNode:
    def __init__(self, host, port, max_peers=5):
        self.host = host
        self.port = port
        self.max_peers = max_peers
        self.peers_incoming = set()
        self.peers_outgoing = set()

    async def start(self):
        server = await asyncio.start_server(self.handle_connection, self.host, self.port)
        print(f"Node listening on {self.host}:{self.port}")

        async with server:
            await server.serve_forever()

    async def handle_connection(self, reader, writer):

        peer_host, peer_port = writer.get_extra_info('peername')

        print(f"Incoming connection from {peer_host}:{peer_port}")

        if len(self.peers_incoming) >= self.max_peers:
            print(f"Max peers limit reached for {peer_host}:{peer_port}")
            peer_list = ",".join([f"{socket}" for socket in self.peers_incoming])+"\n"
            writer.write(peer_list.encode())
            await writer.drain()
            writer.close()
            await writer.wait_closed()
            return

        initial_msg = await reader.readline()
        client_socket = initial_msg.decode().strip()
        print(f"Received initial message from {peer_host}:{peer_port}: {client_socket}")
        self.peers_incoming.add(client_socket)
        accept_msg = "ACCEPT\n"
        writer.write(accept_msg.encode())

        while True:
            data = await reader.readline()
            if not data:
                break
            message = data.decode().strip()
            print(f"Received message from {peer_host}:{peer_port}: {message}")

            # TODO Handle the received message here

        print(f"Connection with {peer_host}:{peer_port} closed")
        self.peers_incoming.remove(client_socket)

    async def connect_to_peer(self, peer_host, peer_port):
        try:
            reader, writer = await asyncio.open_connection(peer_host, peer_port)
            peer_msg = ":".join([self.host, self.port])+"\n"
            writer.write(peer_msg.encode())

        except (ConnectionRefusedError, asyncio.TimeoutError):
            print(f"Failed to connect to peer {peer_host}:{peer_port}")
            return

        try:
            data = await reader.readline()
            peer_list = data.decode().strip()

            if peer_list:
                if peer_list == "ACCEPT":
                    self.peers_outgoing.add((peer_host, peer_port))
                    print(f"Connected to peer {peer_host}:{peer_port}")

                    while True:
                        data = await reader.readline()
                        if not data:
                            break
                        message = data.decode().strip()
                        print(f"Received message from {peer_host}:{peer_port}: {message}")

                        # TODO Send & Receive messages here

                else:
                    print(f"Connection refused by peer {peer_host}:{peer_port}. Peer list: {peer_list}")
                    sockets_list = peer_list.split(",")

                    print(sockets_list)

                    # TODO choose which peers from list to connect to
                    # TODO keep track of already connected/tried connections to avoid double tries
                    for socket in sockets_list:
                        socket_host, socket_port = socket.split(":")
                        if (peer_host, peer_port) in self.peers_outgoing:
                            continue
                        asyncio.create_task(self.connect_to_peer(socket_host, socket_port))

        except asyncio.CancelledError:
            raise

        except Exception as e:
            print(f"An error occurred while connecting to peer {peer_host}:{peer_port}: {str(e)}")
            # Handle other potential errors

        finally:
            writer.close()
            await writer.wait_closed()

    async def send_message(self, peer_host, peer_port, message):
        try:
            reader, writer = await asyncio.open_connection(peer_host, peer_port)
            writer.write(message.encode())
            await writer.drain()
            writer.close()
            await writer.wait_closed()
        except (ConnectionRefusedError, asyncio.TimeoutError):
            print(f"Failed to send message to peer {peer_host}:{peer_port}")


class PeerNodeBootstrap(PeerNode):
    def __init__(self, host, port):
        super().__init__(host, port, max_peers=None)
        self.peers_outgoing = None

    async def handle_connection(self, reader, writer):

        peer_host, peer_port = writer.get_extra_info('peername')
        print(f"Incoming connection from {peer_host}:{peer_port}")

        initial_msg = await reader.readline()
        client_socket = initial_msg.decode().strip()
        print(f"Received initial message from {peer_host}:{peer_port}: {client_socket}")

        # TODO select randomly peers?
        peer_list = ",".join([f"{socket}" for socket in self.peers_incoming])+"\n"
        writer.write(peer_list.encode())
        await writer.drain()

        # add newly discovered peer to connections
        if client_socket not in self.peers_incoming:
            self.peers_incoming.add(client_socket)


# Entry point for new Peer class
async def main(port, connecting_ports):

    if not connecting_ports:
        node = PeerNodeBootstrap('localhost', port)
    else:
        node = PeerNode('localhost', port, max_peers=2)

    server_task = asyncio.create_task(node.start())

    # Wait for the servers to start (optional)
    await asyncio.sleep(1)

    # Connect to other peers
    for connecting_port in connecting_ports:
        asyncio.create_task(node.connect_to_peer('localhost', connecting_port))

    # Start sending/receiving messages with connected peers

    await server_task

if __name__ == '__main__':
    port = 8000
    connecting_ports = []

    if len(sys.argv) >= 2:
        port = sys.argv[1]

        for i in range(len(sys.argv) - 2):
            connecting_ports.append((sys.argv[i+2]))

    asyncio.run(main(port, connecting_ports))
