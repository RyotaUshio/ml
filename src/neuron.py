import numpy as np
import pandas as pd
from typing import *


class Neuron:
    def __init__(self, w, f: Callable):
        self.w = w # weight vector
        self.n = w.size # numbers of inputs
        self.f = f # activation function
    
    def __len__(self):
        return self.n

    def u(self, x): # inner potential
        # x: input vector
        try:
            ret = np.dot(self.w, x)
        except ValueError as e:
            print(f"Error from Perceptron.u() :")
            raise e
        else:
            return ret

    def __call__(self, x):
        # x: input vector
        return self.f(self.u(x))


### activation functions

def thd(u): # threshold function
    # u: inner potential
    return np.where((u >= 0), 1, 0)

### trainer
class Trainer:
    def __init__(self, algorithm: Callable):
        self.algorithm = algorithm

    def __call__(self, neuron: Neuron, dataset):
        self.algorithm(neuron, dataset)

### dataset
class DataSet:
    def __init__(self, input, output, x0_included=False):
        if len(input) != len(output):
            raise ValueError("from DataSet.__init__(): Invalid-sized array")
        if x0_included:
            self.input = input
        else:
            self.input = np.empty((len(input), len(input[0])+1))
            for i, data in enumerate(input):
                self.input[i] = np.insert(data, 0, 1)
        self.output = output

    def __len__(self):
        return len(self.input)

    def items(self):
        return zip(self.input, self.output)


def Simple(nr: Neuron, data, eta=0.01, eps=1e-16, debug_view=False):
    count = 0
    while True:
        i = np.random.randint(0, len(data))
        x = data.input[i]
        d = data.output[i]
        y = nr(x)
        err = y - d
        dw = -eta * err * x
        if debug_view:
            #print(f"#{count}: x = {x}, y = {y}, d = {d}, err = {err}, w = {nr.w}")
            print(f"#{count}: err = {err}, w = {nr.w}")
                    
        if finish(nr, data):
            print(f"finished after {count} loops")
            break
        nr.w += dw
        count += 1


def SGD(nr: Neuron, data, eta=0.01, eps=1e-16, debug_view=False, alpha=1.0):
    # nr.f must be sigmoid
    count = 0
    while True:
        i = np.random.randint(0, len(data))
        x = data.input[i]
        d = data.output[i]
        y = nr(x)
        err = y - d
        delta = err * y * (1 - y) * alpha
        dw = -eta * delta * x
        if debug_view:
            print(f"#{count}: x = {x}, y = {y}, d = {d}, delta = {delta}, w = {nr.w}")
            #print(f"#{count}: delta = {delta}, w = {nr.w}")
                    
        if finish(nr, data):
            print(f"finished after {count} loops")
            break
        nr.w += dw
        count += 1

        

def test(nr: Neuron, data: DataSet):
    wrong = 0
    for IN, OUT in data.items():
        if (nr(IN) - 0.5) * (OUT - 0.5) < 0:#nr(IN) != OUT:
            wrong += 1
    print(f"correct: {len(data) - wrong}/{len(data)}")

def finish(nr: Neuron, data: DataSet):
    for IN, OUT in data.items():
        if (nr(IN) - 0.5) * (OUT - 0.5) < 0:#nr(IN) != OUT:
            return False
    return True
    


def txt2array(filename, height, width):
    df = pd.read_csv(filename,
                     sep=" ",
                     header=None)
    df.dropna(how="any", axis=0, inplace=True)
    df = df.astype(np.float64)
    return df.values.reshape((int(df.size/(height*width)), height*width))
    
