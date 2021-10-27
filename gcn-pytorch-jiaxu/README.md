# Graph Convolutional Network (GCN) Implementation

## Manual & Notes
![GCN](figure.png)
### Data Preprocessing

`idx_features_labels` is used to store and read the contents of the `cora/cora.content` file. The method `genfromtxt()` in numpy is to quickly convert the text content saved in txt to an array in numpy. The format of the file data is id-features-labels, so respectively `idx_features_labels[:, 0]` (id), `idx_features_labels[:, 1:-1]` (features), and `idx_features_labels[:, -1]` (labels).

Here we give a preview of `idx_features_labels` array:
```
[['31336' '0' '0' ... '0' '0' 'Neural_Networks']
 ['1061127' '0' '0' ... '0' '0' 'Rule_Learning']
 ['1106406' '0' '0' ... '0' '0' 'Reinforcement_Learning']
 ...
 ['1128978' '0' '0' ... '0' '0' 'Genetic_Algorithms']
 ['117328' '0' '0' ... '0' '0' 'Case_Based']
 ['24043' '0' '0' ... '0' '0' 'Neural_Networks']]
```

The labels (i.e. the last column) is encoded into one-hot vector fashion. Since the nodes in the file are not in order, a hash table `idx_map` with number `0-(node_size - 1)` is created, and each item in the hash table is `id: number`, i.e., the node `id` corresponds to `number`.

`edges_unordered` is the edge table file and is an array of `(edge_num, 2)`, with each row representing the idx of two vertces of an edge. We give a preview:
```
[[     35    1033]
 [     35  103482]
 [     35  103515]
 ...
 [ 853118 1140289]
 [ 853155  853118]
 [ 954315 1155073]]
```
Since ids of the endpoints are stored in `edges_unordered`, and the ids of each item should be replaced with a number, we use idx in `idx_map` as the key to find the number of the corresponding node, reshape it into an array with the same shape as `edges_unordered`, giving `edges`:
```
[[ 163  402]
 [ 163  659]
 [ 163 1696]
 ...
 [1887 2258]
 [1902 1887]
 [ 837 1686]]
```
The `normalize()` method is used to normalize the feature matrix `features` and the adjacency matrix `adj`, respectively. First, we sum up each row to get `rowsum`; find the reciprocal to get `r_inv`; if a row is all 0, `r_inv` will be equal to INF, and `r_inv` of these rows will be set to 0; construct a diagonal matrix with diagonal elements of `r_inv`; we use the **dot product** of the diagonal matrix and the original matrix to perform normalization, i.e. each element of the original matrix will be multiplied by the corresponding `r_inv`.

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

1. GitHub repo, <a href="https://github.com/tkipf/pygcn">Graph Convolutional Networks in PyTorch</a>
2. Thomas Kipf, blog, <a href="https://tkipf.github.io/graph-convolutional-networks/">Graph Convolutional Networks (2016)</a>
3. Paper, [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907)