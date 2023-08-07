# PeerToPeerFederatedLearning
Start simulation by running:
`python SimulationML.py [opt.: timeout seconds] [opt.: num of standard nodes] [opt.: num of connections] [opt.: start port number] [opt.: type, either avg or max]`.
Example:
`python SImulationML.py 600 50 10 8000 avg`
## Parameters
Standard parameters are:

Timeout: 300 seconds\
Bootstrap nodes: 6\
Nodes: 50

Number of max connections (outgoing): 10\
Number of max connections (incoming): 15 (always incoming+5)

Port range: 8000-8055 on localhost

ML Type: avg\
Epochs: 2\
Train samples per Peer: 512\
Test samples per Peer: 512\
Classes per Peer: 3

Results are saved in files with timestamp under ./results

All parameters can be changed in the main of SimulationML.py
