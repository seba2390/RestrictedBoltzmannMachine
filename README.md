# RestrictedBoltzmannMachine
An implementation of a (binary) Restricted Boltzmann Machine (bRBM) in Python using K-step Contrastive divergence for
training. The bRBM is applied on the infamous MNIST handwritten digits dataset, with the 28x28 single-channel arrays 
of digits being flattened and 'binarized' according to floor(array/max(array)), prior to training.

