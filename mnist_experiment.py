from mnist import *
import pandas as pd


eta_ = [0.0005, 0.005, 0.05, 0.5, 1.0, 5.0]

# 学習係数の影響
def learn_rate(etas=eta_):
    for eta in etas:
        mnist(eta=eta, max_iter=5, eps=0.001, log_cond=lambda m, i: i%5000==0)[1].to_file(f"./log/eta_{eta}.log")

# fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
# for eta in eta_:
#     log = nn.logger.from_file(f"./log/eta_{eta}.log")
#     log.show(ax=ax, label=f"$\\eta={eta}$")

# fig.legend(bbox_to_anchor=(0.08,1), loc="upper left")


# ノイズ耐性
def noise(n=None):
    for n_layer in [2, 3, 4, 5, 10, 20]:
        if n is not None:
            if n != n_layer:
                continue
        for n_neuron in [5, 10, 100, 200, 500, 1000]:#, 5000]:
            for prob in np.arange(0.0, 0.30, 0.05):
                print(f"n_neuron={n_neuron}, n_layer={n_layer}, prob={prob}")
                mnist(prob=prob, n_neuron=n_neuron, n_layer=n_layer, max_iter=10, eta=0.2)[1].to_file(f"./log/noise_{n_neuron}neuron_{n_layer}layer_prob{int(100*prob)}pct.log")

def noise_log():
    df = pd.DataFrame(columns=[5, 10, 100, 200, 500, 1000, 5000], index=[1, 2, 3, 4, 5, 10, 20], dtype=str)
    df.loc[:, :] = ''
    for n_layer in [1, 2, 3, 4, 5, 10, 20]:
        for n_neuron in [5, 10, 100, 200, 500, 1000, 5000]:
            for prob in np.arange(0.0, 0.30, 0.05):
                try:
                    log = nn.logger.from_file(f"./log/noise_{n_neuron}neuron_{n_layer}layer_prob{int(100*prob)}pct.log")
                    print(f"n_layer={n_layer}, n_neuron={n_neuron}, prob={prob}:\trate={log.rate}, time={log.time}")
                    if log.time is None:
                        t = "-"
                    else:
                        t = int(log.time)
                    if log.rate >= 95.0:
                        df.loc[n_layer, n_neuron] = f"{int(prob*100)} % ({log.rate:.1f} %, {t} sec)"
                except Exception as e:
                    print(e)
    return df

net, log = mnist(n_layer=1, n_neuron=100, eta=0.2)


# 初期重みの影響

# 早期終了


# 中間層?
#learn_rate()
