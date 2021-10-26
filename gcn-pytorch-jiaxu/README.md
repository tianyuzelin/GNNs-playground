# GCN Implementation

## Manual & Notes
![GCN](figure.png)
### Data Preprocessing


### Graph Convolution Definition

The `__init__` section initializes some of the parameters, including the input and output dimensions, and initializes the weights for each layer.

The `forward()` method illustrates the operation of each layer on the data, which is first multiplying the input feature matrix with the weight matrix (support), and followed by multiplying the normalized adjacency matrix on the left (output). Since the adjacency matrix is stored with a sparse matrix, it is different from the previous normal `torch.mm`, `torch.spmm` means that sparse_tensor is multiplied with dense_tensor.

### Model Definition

The GCN consists of two GraphConvolution layers, and the `forward(self, x, adj)` method of the GCN corresponds to the input of the feature and adjacency matrix, respectively. The final output is the result of the `log_softmax` transform of the output layer.

### Loss Func

Classification loss, with cross-entropy. Since `log_softmax` has been used in the calculation of output, the loss function used here is `NLLloss`, if the log operation is not added, here we have to use `CrossEntropyLoss`. Regarding the difference between these two losses, the cross entropy one is simply integrating the two operations in nll (i.e. taking negative log, multipliy by label, then average).

### Training & Testing

`model.train()` tells your model that you are training the model. So effectively layers like dropout, batchnorm etc. which behave different on the train and test procedures know what is going on and hence can behave accordingly.

More details: It sets the mode to train. You can call either `model.eval()` or `model.train(mode=False)` to tell that you are testing. It is somewhat intuitive to expect train function to train model but it does not do that. It just sets the mode.

## Dependencies
torch, numpy, scipy

## Reference