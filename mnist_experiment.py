from mnist import *

eta_ = [0.0005, 0.005, 0.05, 0.5, 1.0, 5.0]
mnist = image_classifier

# 学習係数の影響
def learn_rate(etas=eta_):
    for eta in etas:
        mnist(eta=eta, max_epoch=5, eps=0.001, log_cond=lambda count: count%5000==0)[1].to_file(f"./log/eta_{eta}.log")

        
# ノイズ耐性
def noise(n=None):
    for n_layer in [1, 2, 3, 4, 5]:
        if n is not None:
            if n != n_layer:
                continue
        for n_neuron in [5, 10, 100, 200, 500, 1000]:
            for prob in np.arange(0.0, 0.30, 0.05):
                print(f"n_neuron={n_neuron}, n_layer={n_layer}, prob={prob}")
                mnist(prob=prob, n_neuron=n_neuron, n_layer=n_layer, max_epoch=10, eta=0.2)[1].to_file(f"./log/noise_{n_neuron}neuron_{n_layer}layer_prob{int(100*prob)}pct.log")

def noise_log():
    n_layers = [1, 2, 3, 4, 5]
    n_neurons = [5, 10, 100, 200, 500, 1000]
    probs = np.arange(0.0, 0.30, 0.05)

    # fix n_layer
    for n_layer in n_layers:
        fig, ax = plt.subplots()
        ax.set(title=f"{n_layer} Hidden Layers",
               xlabel="Noise Frequency",
               ylabel="Accuracy [%]",
               ylim=(0,100)
        )
        
        for n_neuron in n_neurons:
            rates = []
            for prob in probs:
                try:
                    log = nn.logger.from_file(f"./log/noise_{n_neuron}neuron_{n_layer}layer_prob{int(100*prob)}pct.log")
                    print(f"n_layer={n_layer}, n_neuron={n_neuron}, prob={prob}:\trate={log.rate}, time={log.time}")
                    rates.append(log.rate)
                    
                except Exception as e:
                    print(e)

            try:
                ax.plot(probs, rates, label=f"{n_neuron} neurons/1 hidden layer")
            except Exception as e:
                print(e)
        fig.legend()
        fig.savefig(f"log/{n_layer}layer.png")

    # fix n_neuron
    for n_neuron in n_neurons:
        fig, ax = plt.subplots()
        ax.set(title=f"{n_neuron} Neurons per a Hidden Layer",
               xlabel="Noise Frequency",
               ylabel="Accuracy [%]",
               ylim=(0,100)
        )
        
        for n_layer in n_layers:
            rates = []
            for prob in probs:
                try:
                    log = nn.logger.from_file(f"./log/noise_{n_neuron}neuron_{n_layer}layer_prob{int(100*prob)}pct.log")
                    print(f"n_layer={n_layer}, n_neuron={n_neuron}, prob={prob}:\trate={log.rate}, time={log.time}")
                    rates.append(log.rate)
                    
                except Exception as e:
                    print(e)

            try:
                ax.plot(probs, rates, label=f"{n_layer} hidden layers")
            except Exception as e:
                print(e)
        fig.legend()
        fig.savefig(f"log/{n_neuron}neuron.png")

