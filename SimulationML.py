import sys
import asyncio
import random
import time
import numpy as np
import pandas as pd

import Peer
import networkx as nx
import matplotlib.pyplot as plt

SIMULATION_TIME = 300


class NetworkSimulator:
    def __init__(self):
        self.nodes = []

    async def simulate_network(self):
        tasks = []
        # Start the nodes in parallel
        for (node, bootstrap_port) in self.nodes:
            server_task = asyncio.create_task(node.start_server())
            await asyncio.sleep(0.5)  # wait for server to start listening
            tasks.append(server_task)
            if bootstrap_port is not None:  # connect to bootstrap host
                node.bootstrap_port = bootstrap_port
                connect_task = asyncio.create_task(node.connect_to_peer('localhost', bootstrap_port, True))
                tasks.append(connect_task)

            # tasks.append(query_loop_task)

        # Wait for all connections to establish
        try:
            # Wait for all connections to establish within the timeout
            await asyncio.wait_for(asyncio.gather(*tasks), SIMULATION_TIME)
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
                         'connected_classes': node.connected_classes,
                         'val_local': node.model.val_results,
                         'val_global': node.model.validate_global()}
            connection_info.append(node_info)
        return connection_info


def aggregate_data(connection_info, index_non_bootstrap):
    graph = nx.DiGraph()
    incoming_connections_aggr = {key: 0 for key in range(15)}
    outgoing_connections_aggr = {key: 0 for key in range(15)}
    classes_owned_aggr = {key: 0 for key in range(10)}
    classes_connected_aggr = {key: 0 for key in range(10)}
    val_local_aggr = []
    val_global_aggr = []

    color_map = []

    for node_info in connection_info:

        host, port = node_info['node_id']
        peers = node_info['peers']
        connections = node_info['connections']
        classes_owned = node_info['classes']
        classes_connected = node_info['connected_classes']
        val_local_aggr.append(node_info['val_local'])
        val_global_aggr.append(node_info['val_global'])

        # Collect aggregated data only for non-bootstrap peers
        if port >= index_non_bootstrap:
            # Aggregate in- and outgoing connections info
            if peers is not None and connections is not None and classes_connected is not None and classes_owned is not None:
                if len(peers) < len(incoming_connections_aggr):
                    incoming_connections_aggr[len(peers)] += 1
                if len(connections) < len(outgoing_connections_aggr):
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

        # Build graph (largest component)
        if len(connections) > 0:
            graph.add_node(port)
        for (conn_host, conn_port) in connections:
            graph.add_edge(port, conn_port)

    data_aggregated = {
        'incoming_conn': incoming_connections_aggr,
        'outgoing_conn': outgoing_connections_aggr,
        'classes_owned': classes_owned_aggr,
        'classes_conn': classes_connected_aggr,
        'val_local': val_local_aggr,
        'val_global': val_global_aggr
    }

    current_timestamp = time.strftime('%Y_%m_%d-%H_%M_%S')

    nx.write_adjlist(graph, f"./results/graph_{current_timestamp}.adjlist")

    create_plot(graph, color_map, data_aggregated, current_timestamp)
    save_data(graph, data_aggregated, current_timestamp)


def create_plot(graph, color_map, data_aggregated, timestamp):
    # Retrieve data for charts from dict
    incoming_connections_aggr = data_aggregated['incoming_conn']
    outgoing_connections_aggr = data_aggregated['outgoing_conn']
    classes_owned_aggr = data_aggregated['classes_owned']
    classes_connected_aggr = data_aggregated['classes_conn']
    val_local = data_aggregated['val_local']
    val_global = data_aggregated['val_global']

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    axes[0, 0].bar(incoming_connections_aggr.keys(), incoming_connections_aggr.values())
    axes[0, 0].set_title('Degree Distribution of incoming connections')
    axes[0, 0].set_xlabel('# Connections')
    axes[0, 0].set_ylabel('# Nodes')

    axes[0, 1].bar(outgoing_connections_aggr.keys(), outgoing_connections_aggr.values())
    axes[0, 1].set_title('Degree Distribution of outgoing connections')
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

    # Plot the NetworkX graph
    axes[0, 2].axis('off')
    axes[0, 2].set_title('Network Diagram')
    pos = nx.spring_layout(graph)
    nx.draw_networkx(graph, pos, ax=axes[0, 2], with_labels=True, arrows=True,
                     node_color=color_map, edge_color='gray', node_size=50, font_size=6)

    # Plot all local lines
    for data in val_local:
        axes[1, 2].plot(data, color='b', alpha=0.5, linewidth=0.5)

    # Plot the averaged line
    axes[1, 2].plot(average_curve(val_local), color='r', linewidth=2, label="avg. local validity")

    axes[1, 2].set_title('Local validation results (avg)')
    axes[1, 2].set_xlabel('Step')
    axes[1, 2].set_ylabel('Accuracy')

    axes[1, 2].axhline(y=np.average(val_global), color='g', linestyle='dashed', label="global validity")
    axes[1, 2].legend(bbox_to_anchor=(1.0, 1), loc='upper center')

    # Adjust spacing between subplots
    fig.tight_layout()

    plt.savefig(f"./results/SimulationML_{timestamp}.png")


def save_data(graph, data_aggregated, timestamp):
    val_local = data_aggregated['val_local']
    val_global = data_aggregated['val_global']

    dataframe_dict = {'num_nodes': [num_nodes], 'num_bootstrap_nodes': [num_bootstrap_nodes + 1],
                      'connections': [num_connections],
                      'classes_per_node': [classes_per_node], 'num_samples': [num_samples],
                      'ML_combining_type': [ml_type], 'num_epochs': [num_epochs]}

    if nx.is_connected(graph.to_undirected()):
        dataframe_dict['network_diameter'] = [nx.diameter(graph.to_undirected())]
        dataframe_dict['avg_path_length'] = [nx.average_shortest_path_length(graph.to_undirected())]
    else:
        dataframe_dict['Unconnected'] = ['yes']

    results_dict = {}

    results_dict.fromkeys(list(i for i in range(num_nodes + num_bootstrap_nodes + 1)))

    # set avg values
    val_local_avg = average_curve(val_local)
    results_avg = np.insert(val_local_avg, 0, np.average(val_global))

    results_dict['validation'] = ["global"] + [f"model_{model_index}" for model_index in range(len(val_local_avg))]
    results_dict['avg'] = results_avg

    for i, local_acc in enumerate(val_local):
        local_acc_padded = local_acc + [local_acc[-1]] * (len(val_local_avg) - len(local_acc))
        results_dict[i] = [val_global[i]] + local_acc_padded

    df_settings = pd.DataFrame.from_dict(dataframe_dict)
    df_results = pd.DataFrame.from_dict(results_dict)
    df_settings.to_csv(f"./results/data_settings_{timestamp}.csv")
    df_results.to_csv(f"./results/data_results_{timestamp}.csv")

    # Print metrics to console
    print("\nSETTINGS:")
    print(
        f"Nodes: {num_nodes}, Bootstrap_nodes: {num_bootstrap_nodes + 1}, Classes per node: {classes_per_node}, Connections: {num_connections}")
    print(
        f"Combining Type: {ml_type}, Epochs: {num_epochs}, Train samples: {num_training_samples}, Test samples: {num_test_samples}")

    print("\nNETWORK METRICS:")
    if nx.is_connected(graph.to_undirected()):
        print(f"Network diameter: {nx.diameter(graph.to_undirected()):.2f}")
        print(f"Average path length: {nx.average_shortest_path_length(graph):.2f}")
    else:
        print(f"Graph is unconnected")

    print("\nML Metrics:")
    print(f"Global validation results: {val_global}")
    print(f"Global validation avg.: {np.average(val_global):.2f}, Global validation stddev: {np.std(val_global):.2f}")


def average_curve(data_arrays):
    max_length = max(len(data) for data in data_arrays)

    # Initialize the sum array
    sum_array = np.zeros(max_length)
    count_array = np.zeros(max_length)

    # Iterate over each peer's array and add their data points to the sum array
    for data in data_arrays:
        sum_array[:len(data)] += data
        count_array[:len(data)] += 1

    # Divide each position in the sum array by the number of peers contributing to that position
    averaged_curve = np.divide(sum_array, count_array, out=np.zeros_like(sum_array), where=count_array != 0)
    return averaged_curve


# Example usage
if __name__ == '__main__':
    max_classes = 10  # amount of classes for simulation
    classes_per_node = 5  # amount of classes assigned to each node
    num_nodes = 50  # amount of standard nodes
    num_bootstrap_nodes = 5  # amount of bootstrap nodes

    ml_type = "avg"
    num_epochs = 2
    num_samples = 512
    num_training_samples = num_samples
    num_test_samples = num_samples

    network = NetworkSimulator()

    num_connections = 10
    port_number = 8000

    if len(sys.argv) > 1:
        SIMULATION_TIME = int(sys.argv[1])
    if len(sys.argv) > 2:
        num_nodes = int(sys.argv[2])
    if len(sys.argv) > 3:
        num_connections = int(sys.argv[3])
    if len(sys.argv) > 4:
        port_number = int(sys.argv[4])
    if len(sys.argv) > 5:
        ml_type = sys.argv[5]

    num_peers = 15

    # Add initial (empty) bootstrap node
    node = Peer.PeerNode('localhost',
                         port_number,
                         max_peers=num_peers,
                         max_connections=num_connections,
                         num_classes=classes_per_node,
                         max_classes=max_classes,
                         ml_type=ml_type,
                         num_training_samples=num_training_samples,
                         num_test_samples=num_test_samples)
    network.add_node(node, None)

    # Add further bootstrap nodes (optional, only for num_bootstrap_nodes>1)
    for bootstrap_port_index in range(num_bootstrap_nodes - 1):
        bootstrap_port_number = port_number + 1 + bootstrap_port_index
        node = Peer.PeerNode('localhost',
                             bootstrap_port_number,
                             max_peers=num_peers,
                             max_connections=num_connections,
                             num_classes=classes_per_node,
                             max_classes=max_classes,
                             ml_type=ml_type,
                             num_training_samples=num_training_samples,
                             num_test_samples=num_test_samples)
        network.add_node(node, port_number)  # Add with connection to empty bootstrap node

    # create nodes
    for new_port_index in range(num_nodes):
        new_port_number = port_number + num_bootstrap_nodes + 1 + new_port_index
        node = Peer.PeerNode('localhost',
                             new_port_number,
                             max_peers=num_peers,
                             max_connections=num_connections,
                             num_classes=classes_per_node,
                             max_classes=max_classes,
                             ml_type=ml_type,
                             num_training_samples=num_training_samples,
                             num_test_samples=num_test_samples)

        network.add_node(node, random.randint(port_number, port_number + num_bootstrap_nodes - 1))

    network.start_simulation()  # Start the simulation
    connection_info = network.get_connection_info()  # Get distribution information from established network

    aggregate_data(connection_info, port_number + num_bootstrap_nodes + 1)  # Build bar charts and graph
