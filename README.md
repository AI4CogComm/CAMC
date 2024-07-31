# CAMC
Source codes of the article: 

P. Dong, C. He, S. Gao, F. Zhou and Q. Wu, "Edge Learning Based Collaborative Automatic Modulation Classification for Hierarchical Cognitive Radio Networks," IEEE Internet of Things Journal, to be published. 

Please cite this paper when using the codes.

# Instructions

Reading the below sections in the written order will help better understand all the codes.

 ## Python

**rmldataset2016.py**
This code is used for data preprocessing, including converting IQ signals to AP format, normalization, and other tasks.

**mltools.py**
This code is a tool class used to draw figures.

**rmlmodels**
This code defines the network structure of the model, detailing the layers, activation functions, and connections used to build the neural network.

**C-AMC.py**
This script is designed for the C-AMC model workflow. It includes the processes for training and testing the model, and ultimately generates a graph displaying classification accuracy and a confusion matrix for performance evaluation.

**Pruned**
This folder contains code for pruning various models, specifically SSCNet, MCNet, and C-AMC.

**Quantized**
This folder contains code for quantizing various pruned models, in particular SSCNet, MCNet, and C-AMC.

**Transfer Learning**
This folder contains code for transfer learning using the RML2016.10a and RML2018.01a datasets.

# Environment
These models are implemented in Keras, and the environment setting is:

-   Python 3.7.16
-   TensorFlow-gpu 2.7.0
