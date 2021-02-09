# ml
A pattern recognition & machine learning package for Python. It provides:

1. Classification algorithms, such as
the k-nearest neighbors algorithm, a classifier by generative approaches, the simple perceptron, classification with MLP : multilayer perceptron
1. Clustering algorithms, such as
K-means, competitive learning, the EM algorithm(, mean shift)
1. Regression analysis including the standard linear regression, Ridge regression, (lasso, ...)
1. Feature extraction & dimensionality reduction, such as
PCA : principal component analysis, whitening transformation, LDA : linear discriminant analysis and autoencoders (simple & stacked).

## Author
* **Ryota Ushio** - https://github.com/RyotaUshio

## Examples
### Autoencoder
An autoencoder is a type of neural networks.
Typically, it constists of same-sized input & output layers and some hidden layers with less number of units and is trained to be able to restore the input.
As the result, the hidden layer's activation pattern, which has less dimensions than the input, can be interpreted to hold "essential" information about the input patter in a sense.
Therefore, it is used for feature extraction or dimensionality reduction.

Here's an example using the MNIST handwritten digits dataset. See `tests/test_ae.py` for the details.

| Input | Output | 
| --- | ---|
![](https://github.com/RyotaUshio/ml/blob/main/fig/ae_original3.png) | ![](https://github.com/RyotaUshio/ml/blob/main/fig/ae_restored3.png)
![](https://github.com/RyotaUshio/ml/blob/main/fig/ae_original5.png) | ![](https://github.com/RyotaUshio/ml/blob/main/fig/ae_restored5.png)

### Clustering algorithms

Here're some examples for clustering algorithms. See `tests/test_cluster2.py` for the details.

* Input

<img src="https://github.com/RyotaUshio/ml/blob/main/fig/test_cluster2_original.png" height="30%" width="30%">

* Outputs

| K-means | Competitive Learning | EM Algorithm |
| --- | --- | --- |
| ![](https://github.com/RyotaUshio/ml/blob/main/fig/test_cluster2_kmeans.png) | ![](https://github.com/RyotaUshio/ml/blob/main/fig/test_cluster2_competitive.png) | ![](https://github.com/RyotaUshio/ml/blob/main/fig/test_cluster2_em.png)
