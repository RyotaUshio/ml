import nn
import mnist
import numpy as np


mnist.load()

# 中間層のニューロン数をいろいろ変えてAIC, BICを計算してみる。
n_hidden_unit = [5, 10, 20, 50, 100, 200, 500, 1000, 2000]
aic = []
bic = []
nets = []

for n in n_hidden_unit:
    net = mnist.image_classifier(X_test=mnist.X_test, T_test=mnist.T_test, hidden_shape=[n], max_epoch=1000, batch_size=200)
    nets.append(net)
    aic.append(net.log.AIC)
    bic.append(net.log.BIC)

    
