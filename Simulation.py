import sys
import asyncio
import random
import Peer
import networkx as nx
import matplotlib.pyplot as plt


class NetworkSimulator:
    def __init__(self):
        self.nodes = []

    async def simulate_network(self):
        tasks = []
        # Start the nodes in parallel
        for (node, bootstrap_port) in self.nodes:
            listen_task = asyncio.create_task(node.start_server())
            await asyncio.sleep(0.5)                                    # wait for server to start listening
            tasks.append(listen_task)
            if bootstrap_port is not None:                              # connect to bootstrap host
                connect_task = asyncio.create_task(node.connect_to_peer('localhost', bootstrap_port, True))
                tasks.append(connect_task)

        # Wait for all connections to establish
        try:
            # Wait for all connections to establish within the timeout
            await asyncio.wait_for(asyncio.gather(*tasks), 1)
        except asyncio.TimeoutError:
            pass

    def add_node(self, node, bootstrap_port):
        self.nodes.append((node, bootstrap_port))

    def start_simulation(self):
        asyncio.run(self.simulate_network())

    def get_connection_info(self):
        connection_info = []
        for (node, _) in self.nodes:
            node_info = {'node_id': (node.host, node.port),
                         'classes': node.classes,
                         'peers': list(node.peers.keys()),
                         'connections': list(node.connections.keys()),
                         'connected_classes': node.connected_classes}
            connection_info.append(node_info)
        return connection_info


def aggregate_data(connection_info, index_non_bootstrap):
    graph = nx.DiGraph()
    incoming_connections_aggr = {key: 0 for key in range(6)}
    outgoing_connections_aggr = {key: 0 for key in range(6)}
    classes_owned_aggr = {key: 0 for key in range(10)}
    classes_connected_aggr = {key: 0 for key in range(10)}

    color_map = []

    for node_info in connection_info:

        host, port = node_info['node_id']
        peers = node_info['peers']
        connections = node_info['connections']
        classes_owned = node_info['classes']
        classes_connected = node_info['connected_classes']

        # Collect aggregated data only for non-bootstrap peers
        if port >= index_non_bootstrap:
            # Aggregate in- and outgoing connections info
            incoming_connections_aggr[len(peers)] += 1
            outgoing_connections_aggr[len(connections)] += 1

            # Aggregate owned classes info
            for class_owned in classes_owned:
                classes_owned_aggr[class_owned] += 1

            # Aggregate connected classes info
            for class_connected, connected_amount in classes_connected.items():
                classes_connected_aggr[class_connected] += connected_amount

            color_map.append('lightblue')
        else:
            color_map.append('red')

        # Build graph
        graph.add_node(port)
        for (conn_host, conn_port) in connections:
            graph.add_edge(port, conn_port)

        data_aggregated = {
            'incoming_conn': incoming_connections_aggr,
            'outgoing_conn': outgoing_connections_aggr,
            'classes_owned': classes_owned_aggr,
            'classes_conn': classes_connected_aggr
        }

    return create_plot(graph, color_map, data_aggregated)


def create_plot(graph, color_map, data_aggregated):

    # Retrieve data for charts from dict
    incoming_connections_aggr = data_aggregated['incoming_conn']
    outgoing_connections_aggr = data_aggregated['outgoing_conn']
    classes_owned_aggr = data_aggregated['classes_owned']
    classes_connected_aggr = data_aggregated['classes_conn']

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    axes[0, 0].bar(incoming_connections_aggr.keys(), incoming_connections_aggr.values())
    axes[0, 0].set_title('Distribution of incoming connections')
    axes[0, 0].set_xlabel('# Connections')
    axes[0, 0].set_ylabel('# Nodes')

    axes[0, 1].bar(outgoing_connections_aggr.keys(), outgoing_connections_aggr.values())
    axes[0, 1].set_title('Distribution of outgoing connections')
    axes[0, 1].set_xlabel('# Connections')
    axes[0, 1].set_ylabel('# Nodes')

    axes[1, 0].bar(classes_owned_aggr.keys(), classes_owned_aggr.values())
    axes[1, 0].set_title('Distribution of owned classes')
    axes[1, 0].set_xlabel('Class')
    axes[1, 0].set_ylabel('Frequency')

    axes[1, 1].bar(classes_connected_aggr.keys(), classes_connected_aggr.values())
    axes[1, 1].set_title('Distribution of connected classes')
    axes[1, 1].set_xlabel('Class')
    axes[1, 1].set_ylabel('Frequency')

    nx.draw_networkx(graph, with_labels=True, arrows=True, node_color=color_map, edge_color='gray', node_size=50,
                     font_size=6)

    # Plot the NetworkX graph
    axes[0, 2].set_title('Network Diagram')
    pos = nx.spring_layout(graph)
    nx.draw_networkx(graph, pos, ax=axes[0, 2], with_labels=True, arrows=True,
                     node_color=color_map, edge_color='gray', node_size=50, font_size=6)

    # Remove unused subplots
    fig.delaxes(axes[1, 2])

    # Adjust spacing between subplots
    fig.tight_layout()

    # Show the plot
    plt.show()


# Example usage
if __name__ == '__main__':
    max_classes = 10            # amount of classes for simulation
    classes_per_node = 3        # amount of classes assigned to each node
    num_nodes = 100             # amount of standard nodes
    num_bootstrap_nodes = 5     # amount of bootstrap nodes
    network = NetworkSimulator()

    port_number = 8000

    # Add initial (empty) bootstrap node
    node = Peer.PeerNode('localhost', port_number)
    network.add_node(node, None)

    # Add further bootstrap nodes (optional, only for num_bootstrap_nodes>1)
    for bootstrap_port_index in range(num_bootstrap_nodes-1):
        bootstrap_port_number = port_number+1+bootstrap_port_index
        node = Peer.PeerNode('localhost',
                             bootstrap_port_number,
                             max_peers=5,
                             max_connections=5,
                             max_classes=max_classes)
        node.classes = random.sample(range(max_classes), classes_per_node)          # Set random classes
        network.add_node(node, port_number)                     # Add with connection to empty bootstrap node

    # create nodes
    for new_port_index in range(num_nodes):
        new_port_number = port_number+num_bootstrap_nodes+1+new_port_index
        node = Peer.PeerNode('localhost',
                             new_port_number,
                             max_peers=5,
                             max_connections=5,
                             max_classes=max_classes)
        node.classes = random.sample(range(max_classes), classes_per_node)          # Set random classes
        # Add to network with connection to any random bootstrap node
        network.add_node(node, random.randint(port_number, port_number+num_bootstrap_nodes-1))

    network.start_simulation()                          # Start the simulation
    connection_info = network.get_connection_info()     # Get distribution information from established network

    aggregate_data(connection_info, port_number+num_bootstrap_nodes+1)      # Build bar charts and graph
