# Neural Peer

## What is Neural Peer?

## Content of the Repository 

## Technologies

## Dependencies

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
