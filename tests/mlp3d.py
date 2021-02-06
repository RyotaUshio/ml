import numpy as np
import matplotlib.pyplot as plt
import ml

# データセットの読み込み
(X_train, T_train), (X_test, T_test) = ml.load_data('mnist')

fig = plt.figure()
plt.ion()
# 訓練中の各エポックで、最後の隠れ層の出力を図示するためのコールバック関数.
def callback(net):
    net.forward_prop(X_train[:5000])
    ml.scatter(net[-2].z.copy(), T_train[:5000], fig=fig)
    plt.pause(0.1)
    
# MLPの訓練開始.
# 訓練用のデータ60000組のうち, 最後の10%を検証用にまわし, 
# 残りの90%で学習させる. 
# 入力層が784ユニット・3つの中間層がそれぞれ500, 200, 3ユニット・出力層が10ユニット.
# 中間層の活性化関数はReLU, ReLU, tanh.
# 学習率0.01, 荷重減衰のパラメータ0.00005で, Momentumを使って重み更新.
# 最大300エポック学習させるが. デフォルトで早期終了(early stopping)がONになっており, 
# 一定期間を越えて検証用データに対するlossに進捗がない場合, 300エポック未満でも終了する. 
net = ml.classify.mlp_classifier.fit(
    X_train[:-6000], T_train[:-6000],
    hidden_shape=[500, 200, 3],
    hidden_act=['ReLU', 'ReLU', 'tanh'],
    X_val=X_train[-6000:], T_val=T_train[-6000:], 
    eta0=1e-2, lamb=0.00005, optimizer='Momentum',
    max_epoch=300, how='stdout', callback=callback
)

# 訓練終了後, 最後の隠れ層の出力を図示する.
# クラスごとに非常にきれいに分離されているのが見て取れる.
# まずは訓練データの場合. 
net.forward_prop(X_train)
X_train_3d = net[-2].z
T_train = ml.utils.vec2label(T_train)
ml.scatter(X_train_3d, T_train)
plt.title("Training data after the last hidden layer: fairly seperated!")
plt.show()
plt.pause(3)

# つぎに, テストデータの場合.
# こちらは訓練データほど綺麗に分かれない. 
net.forward_prop(X_test)
X_test_3d = net[-2].z
T_test = ml.utils.vec2label(T_test)
ml.scatter(X_test_3d, T_test)
plt.title("Test data after the last hidden layer: a little messy...")
plt.show()
plt.pause(3)
