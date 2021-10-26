# GCN Implementation

## Dependencies
- torch
- numpy
- scipy

## Manual & Notes
![GCN](figure.png)

### Graph Convolution Definition

### Model Definition

### Training & Testing
`model.train()` tells your model that you are training the model. So effectively layers like dropout, batchnorm etc. which behave different on the train and test procedures know what is going on and hence can behave accordingly.

More details: It sets the mode to train. You can call either `model.eval()` or `model.train(mode=False)` to tell that you are testing. It is somewhat intuitive to expect train function to train model but it does not do that. It just sets the mode.

### Experiment Data

## Reference