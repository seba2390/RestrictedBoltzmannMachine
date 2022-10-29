# Restricted Boltzmann Machine
An implementation of a (binary) Restricted Boltzmann Machine (bRBM) in Python using K-step Contrastive divergence for
training. The bRBM is applied on the infamous MNIST handwritten digits dataset, with the 28x28 single-channel arrays 
of digits being flattened and 'binarized' according to floor(array/max(array)), prior to training. General theory about the construction and training
of RBM's can be found in [[1]](#1).

#### RBM progress on one digit:
(sampled using Gibbs sampling w. only one repetition at end of each epoch)
![alt text](https://github.com/seba2390/RestrictedBoltzmannMachine/blob/main/media/RBM_capture.gif "bRBM")

## References
<a id="1">[1]</a> 
Fischer, A., Igel, C.
[Training restricted Boltzmann machines - An introduction.](https://doi.org/10.1016/j.patcog.2013.05.025).
Pattern Recognition, Volume 47, Issue 1, 2014, Pages 25-39, ISSN 0031-3203.
