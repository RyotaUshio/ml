import numpy as np
import neuron as neu

A = neu.txt2array("dataset/char/A.txt", 5, 5)
B = neu.txt2array("dataset/char/B.txt", 5, 5)

training_dataset = (
    neu.DataSet(
        np.concatenate([A, B]), 
        np.concatenate([np.zeros(len(A)), np.ones(len(B))])
    )
    )
w0 = np.random.rand(26)
nr = neu.Neuron(w0, neu.thd)
neu.Simple(nr, training_dataset)

A = neu.txt2array("dataset/char/A2.txt", 5, 5)
B = neu.txt2array("dataset/char/B2.txt", 5, 5)
training_dataset = (
    neu.DataSet(
        np.concatenate([A, B]), 
        np.concatenate([np.zeros(len(A)), np.ones(len(B))])
    )
    )
neu.test(nr, training_dataset)

test = np.array([1, 0, 1, 1, 0, 0, 
                 1, 0, 0, 1, 0,
                 1, 1, 1, 1, 0,
                 1, 0, 0, 1, 0, 
                 1, 0, 0, 1, 0])
print(nr(test))


