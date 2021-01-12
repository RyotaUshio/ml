import numpy as np
import dataclasses
import doctest
from typing import Sequence, List, Callable



@dataclasses.dataclass
class layer:
    """
    Feedforward neural networkの各層.
    """
    W: np.ndarray # 前の層から自分への重み行列(結合行列)
    b: np.ndarray # 前の層から自分へのバイアスベクトル
    h: "act_func" # 活性化関数
    size: int     # この層に含まれるニューロンの数
    # この層の各ニューロンの活性(内部ポテンシャル)を並べたベクトル
    u: np.ndarray = dataclasses.field(init=False)
    # 各ニューロンの出力z = h(u)を並べたベクトル
    z: np.ndarray = dataclasses.field(init=False)
    # 各ニューロンの誤差を並べたベクトル
    delta: np.ndarray = dataclasses.field(init=False)
    
    def __post_init__(self):
        self.u = np.zeros((1, self.size))
        self.z = np.zeros((1, self.size))
        self.delta = np.zeros((1, self.size))

        

@dataclasses.dataclass
class mlp:
    """
    多層パーセプトロン(MLP: Multi Layer Perceptron). 
    """
    layers: Sequence[layer] # 各層を表すlayerオブジェクトを並べた配列
    loss: "loss_func" = dataclasses.field(default=None) # 損失関数

    def __post_init__(self):
        if self.loss is None:
            self.loss = self[-1].h.loss
        self.loss.net = self

    def __len__(self):
        return len(self.layers)

    def __getitem__(self, key):
        return self.layers[key]

    def __iter__(self):
        return iter(self.layers)

    def __call__(self, x):
        self.forward_prop(x)
        return self[-1].z.copy()
    
    @classmethod
    def from_num(cls, num: Sequence[int], act_funcs: Sequence["act_func"], sigma:float=0.01, loss=None) -> "mlp":
        """
        各層のニューロン数を表す配列numからlayerオブジェクトとmplオブジェクトを生成する。
        Parameters
        ----------
        num: list
             ネットワークの各層内にいくつのニューロンを生成するか
        sigma: float
             重み行列の初期値を生成するガウス分布の標準偏差
        """
        # 出力層は0オリジンでl = L層目
        L = len(num) - 1
        # 重み行列の各要素は標準偏差sigmaのガウス分布からサンプリングされる
        W = np.array([None] +
                     [np.random.normal(scale=sigma,
                                       size=(num[l-1], num[l]))
                      for l in range(1, L+1)],
                     dtype=object)
        # バイアスは定数初期化.
        b = np.array([None] + [np.zeros((1, num[l])) for l in range(1, L+1)],
                     dtype=object)
        
        layers = [layer(size=num[l], W=W[l], b=b[l], h=act_funcs[l]) for l in range(L+1)]
        return cls(layers, loss=loss)
    
    def forward_prop(self, x:np.ndarray) -> None:
        """順伝播. xは入力ベクトル. ミニバッチでもOK"""
        self[0].z = x
        for l in range(1, len(self)):
            self[l].u = self[l-1].z @ self[l].W + self[l].b
            self[l].z = self[l].h( self[l].u )
            
    def back_prop(self, t:np.ndarray) -> None:
        """教師信号ベクトルtをもとに誤差逆伝播法による学習を行う."""
        # 出力層の誤差はすぐに計算できる.
        self[-1].delta = self.loss.error(t)
        # それを使って誤差を順次前の層へ逆伝播させていく
        for l in range(len(self)-2, 0, -1):
            self[l].delta = self[l].h.grad_from_val(self[l].z) * (self[l+1].delta @ self[l+1].W.T)
            
    def train(self,
              X:np.ndarray,
              T:np.ndarray,
              eta:float=0.005,
              eps:float=-np.inf,
              max_iter:int=10000,
              batch_size=1,
              log_cond:Callable=lambda m, i: i == 0):
        """
        訓練データ集合(X: 入力, T: 出力)をネットワークに学習させる.
        Parameters
        ----------
        X, T: 訓練データ集合
        eta: 学習係数
        eps: 反復停止判定に用いる損失の閾値
        max_iter: 最大反復回数
        batch_size: ミニバッチ学習に用いるミニバッチのサイズ
        log_cond: カウンタ変数m, iがどんな条件を満たしたときに損失関数の値などをlogに記録するか

        Returns
        -------
        log: 途中経過を記録したloggerオブジェクト
        """
        # 入力層が第0層, 出力層が第L層
        L = len(self)-1
        # 途中経過の記録
        log = logger()
        
        try:
            for m in range(max_iter):
                if m > 0 and np.sum(log.loss[-100:]) < eps:
                    break
                #np.random.shuffle(X)
                for i in range(int(np.ceil(len(X)/batch_size))):
                    # ミニバッチ
                    x = X[i*batch_size:(i+1)*batch_size]
                    t = T[i*batch_size:(i+1)*batch_size]
                    N = len(X)
                    # 順伝播
                    self.forward_prop(x)
                    # 逆伝播
                    self.back_prop(t)
                    # 勾配法による重み更新
                    for l in range(1, L+1):
                        dJdW = self[l-1].z.T @ self[l].delta
                        dJdb = np.sum(self[l].delta, axis=0)
                        self[l].W += -eta*dJdW
                        # print(f"delta:{self[l].delta.shape}, z:{self[l-1].z.shape}, b:{self[l].b.shape}, dJdb:{dJdb.shape}, W:{self[l].W.shape}, dJdW:{dJdW.shape}")
                        # print(f"delta:{self[l].delta}, z:{self[l-1].z}, b:{self[l].b}, dJdb:{dJdb}, W:{self[l].W}, dJdW:{dJdW}")
                        self[l].b += -eta*dJdb

                    if log_cond(m, i):
                        log.loss.append(self.loss(t))
                        log.count.append(int(np.ceil(len(X)/batch_size)) * m + i)
                        print(f"m = {m}, i = {i}, J = {log.loss[-1]}")
    
        except KeyboardInterrupt:
            pass
        return log
    
    def test(self, X:np.ndarray, T:np.ndarray, log:"logger"=None) -> None:
        """
        テストデータ集合(X: 入力, T: 出力)を用いて性能を試験し、正解率を返す.
        """
        correct = 0
        for i in range(len(X)):
            if isinstance(self[-1].h, softmax):
                #self.forward_prop(X[[i]])
                if np.argmax(T[i]) == np.argmax(self(X[[i]])):
                    correct += 1

            elif isinstance(self[-1].h, sigmoid):
                ans = 1 if self(X[[i]]) > 0.5 else 0
                if T[i] == ans:
                    correct += 1

        rate = correct / len(X) * 100
        print(f"{rate} % correct")
        if isinstance(log, logger):
            log.rate = rate

            

@dataclasses.dataclass
class act_func:
    """各層の活性化関数"""
    param: float = 1.0
    loss: "loss_func" = dataclasses.field(default=None)

    def __call__(self, u):
        """活性化関数の値h(u)そのもの."""
        raise NotImplementedError

    def grad_from_val(self, z):
        """活性化関数の関数値z = h(u)の関数として導関数h\'(u)の値を計算する."""
        raise NotImplementedError

    
class sigmoid(act_func):
    """シグモイド関数"""
    def __init__(self):
        super().__init__()
        self.loss = cross_entropy()
        
    def __call__(self, u):
        return 0.5 * (1.0 + np.tanh(0.5 * self.param * u))
    
    def grad_from_val(self, z, th=0.001):
        z = np.where(z > 1.0 - th, 1.0 - th, z)
        z = np.where(z < th, th, z)
        return z * (1.0 - z)

    
class linear(act_func):
    """線形関数"""
    def __call__(self, u):
        return u
    
    def grad_from_val(self, z):
        return 1.0

    
class ReLU(act_func):
    """ReLU"""
    def __call__(self, u):
        return np.where(u > 0.0, u, 0.0)
    
    def grad_from_val(self, z):
        return np.where(u > 0.0, 1.0, 0.0)

    
class softmax(act_func):
    """ソフトマックス関数"""
    def __init__(self):
        super().__init__()
        self.loss = mul_cross_entropy()
        
    def __call__(self, u):
        tmp = u - u.max()
        return np.exp(tmp) / np.sum(np.exp(tmp))
    
    def grad_from_val(self, z):
        return Exception("出力層の活性化関数の微分は使わないはず")


    
@dataclasses.dataclass
class loss_func:
    net: mlp = dataclasses.field(default=None)
    
    """損失関数"""    
    def __call__(self, t):
        """
        損失関数の値E(z)そのもの.
        Parameter(s)
        ------------
        net: mlp
        t: 教師信号
        """
        raise NotImplementedError

    def error(self, t):
        """
        出力層の誤差delta = 出力層の内部ポテンシャルuによる微分
        """
        raise NotImplementedError


class mean_square(loss_func):
    def __call__(self, t):
        norms = 0.5 * np.linalg.norm(self.net[-1].z - t, axis=1)
        if norms.shape != (len(t),):
            raise Exception("unexpected shape")
        return np.mean(norms)

    def error(self, t):
        return np.mean(self.net[-1].z - t, axis=0)
    
class cross_entropy(loss_func):
    def __call__(self, t):
        N = len(t)
        z = self.net[-1].z
        return -(t * np.log(z) + (1.0 - t) * np.log(1 - z)).sum() / N
    def error(self, t):
        N = len(t)
        return (self.net[-1].z - t) / N

    
class mul_cross_entropy(cross_entropy):
    def __call__(self, t):
        return -(t * np.log(self.net[-1].z + 1e-7)).mean()


    
@dataclasses.dataclass
class logger:
    """学習経過の記録"""
    loss: Sequence[float] = dataclasses.field(default_factory=list)
    rate: float = dataclasses.field(default=None)
    count: Sequence[int] = dataclasses.field(default_factory=list)
    #cond: Callable = dataclasses.field(default=lambda m, i: i==0)

    def show(self, ax=None, semilog=False, *args, **kwargs):
        if ax is None:
            ax = plt.gca()
        ax.plot(self.count, self.loss, *args, **kwargs)
        ax.set(xlabel="iteration", ylabel="loss")
        if semilog:
            ax.set_yscale("log")

    def to_file(self, fname):
        with open(fname, "w") as f:
            f.write(repr(self))
            
    @classmethod
    def from_file(cls, fname):
        with open(fname) as f:
            return eval("cls" + f.read().replace("logger", ""))
