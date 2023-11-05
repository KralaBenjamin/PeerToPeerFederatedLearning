# Neural Peer

## What is Neural Peer?
Neural Peer is an concept created during the "Peer to Peer" lecture at the summer term of 2023 based on the paper [BrainTorrent | https://arxiv.org/abs/1905.06731], under the guidance of Professor Tschorsch. The idea was developed by brandjak and KralaBenjamin. The primary goal of Neural Peer is to combine the principles of Peer-to-Peer (P2P) networks with the capabilities of Neural Networks to enable decentralized training and collaboration among peers.

In the context of our experiments on Neural Peer, a peer represents an entity with access to a particular subset of knowledge or data. However, due to various constraints such as data protection regulations, these peers are unable to share their data directly. Instead, they can share the weights of their models, which represent the distilled knowledge from their data.


The experiments conducted with Neural Peer revolve around a basic idea of training neural networks in a decentralized manner. The dataset used for this purpose is the popular FashionMNIST, which comprises images of clothing items belonging to different classes. It is also harder to train the classic MNIST. In order to simulate the limited knowledge scenario of each peer, the dataset is divided in such a way that every peer only has access to a fraction of the classes, usually three random classes.

The training process involves every peer independently training its model for one epoch using its local data. All peers use the same NN architecture. You can find the Lenet Architecture [here | https://github.com/KralaBenjamin/PeerToPeerFederatedLearning/blob/76e0528714925715178b066197dfec10a6370e85/ml_class.py#L251C7-L251C12].


Once the training is complete, the peers can send their newly trained models to their neighboring peers.
This happens when a peer ask another peer for data (= model as we are privacy friendly).
After receiving the models from neighbors, each peer either averages the models or selects the best model based on its own test data. This process allows peers to benefit from the knowledge of others without directly accessing their data.



In general the selection of neighbor peers plays a crucial role in the effectiveness of the Neural Peer concept. Every peer aims to maximize its class distribution by carefully choosing neighbors that possess complementary knowledge. By collaborating with peers that have access to different classes, each peer can enhance its model's ability to recognize a wider range of classes, ultimately leading to better performance. Also very important while a good P2P network tries to avoid many connections (reduciong network complexity), in machine learning the peer needs as many peers as possible to gather as much information as possible. This was also one question in our experiments.
It was important to note that this does not happen in the beginning as a peert try to find as many peers as possible.


## Content of the Repository 
* [What is Neural - Peer?](#What-is-Neural-Peer?)
* [Technologies](#Technologies)
* [Dependencies](#Dependencies)
* [Launch](#Launch)
* [Experiments](#Experiments)



## Technologies
- Python
- Peer2Peer
- Pytorch, Machine Learning
- Numpy
- AsyncIO
- matplotlib
- NetworkX
- 

## Dependencies
python=3.11.3
pytorch=2.0.1
matplotlib=3.7.1
networkx=3.1
numpy=1.24.3


## Launch
It is possible build a network by running individual Peers as processes.
In order to gather results from a network in total, there is also a script
to start a pre-determined number of Peers and collects their data during the
time of the experiment.
### Launching an individual Peer instance
In order to instantiate a Peer, the port on which he can listen (on localhost)
must be specified. In order to integrate in a netowrk, an additional port of an
already running instance can be given, which servers as a bootstrap mechanism.
A Peer instance is started with:\
`python Peer.py [listening port] [opt.: bootstrap port]`

After a node is started and integrated in a network, it exchanges model weights
with its neighbours and trains a local model automatically. There is no
termination criterion, so Peers share data and train their models until their
process is terminated by the user.
### Launching an experiment with multiple Peers
To evaluate different approaches and parameters, there is also a script for
starting an experiment with several Peers. The nodes are started one after the
other with increasing port numbers. In the beginning of the experiment, a number
of bootstrap nodes are created to serve as entry points in the network.
After a timeout, all nodes are terminated and their information is collected
and saved.\
Following parameters can be set:
* Timeout to stop experiment (in seconds)
* Number of standard nodes created in the network (excluding bootstrap nodes)
* Number of connections each Peer tries to establish
* Start port (port number, default: 8000), following ports are assigned increasingly
* Type (avg for average of model weights, max for best-performing model)

An experiment is started with:\
`python SimulationML.py [timeout] [num of standard nodes] [num of connections] [start port number] [type]`

After each local training step, the validation is scored with a small local
validation set. Additionally, the final models are evaluated with a large
validation dataset.\
After the timeout has elapsed, the parameters, validation scores and network
properties are saved in csv-filed in */results* with the current
timecode.

## Experiments
We ran multiple experiments to compare the different approaches and test the
influence of the parameters. The result data is saved in subdirectories of
*/results*. When having only data from a number of classes locally available,
we expect the validation score of a model to be at least
*locally available classes / overall classes*. If this value is exceeded, we
assume that the additional information is from exchanging model weights
with neighbours in the network. In most runs, we could observe an increase in
 validation values of around **+0.1**. Two examples can be seen below.

 ![SimulationML_2023_07_15-16_58_12](https://github.com/KralaBenjamin/PeerToPeerFederatedLearning/assets/23062484/480a12fa-f986-4f19-85d3-db98307bd4c1)

 ![SimulationML_2023_07_15-16_57_44](https://github.com/KralaBenjamin/PeerToPeerFederatedLearning/assets/23062484/0f646019-f5d1-4b36-946a-d382181dd0ff)

Results for avg (top) and max (bottom) with 128 samples, 5 connections and 5 classes.
 
 ### Type: avg vs. max
 Regarding validation values, there was no clear trend visible between the
 avg and max approaches. However, when both run with the same timeout, avg
 manages to perform more training steps, because averaging the state weights is
 a faster operation than evaluating each received model to select the best.
 ### Number of samples and epochs
 As default values, each Peer was instantiated with 128 samples which are
 trained for two epochs in the training step. We tried to increase and decrease
 the parameters, but without any noticaeble difference in the validation
 results.
 ### Number of classes
 As expected, the validation for a higher number of local classes is
 proportionally higher. However, the increase of circa +0.1 did not change when
 increasing the amount of classes for each Peer.
