"""コマンドラインから
python3 tests/eval_mlp3d.py no_dropout
または
python3 tests/eval_mlp3d.py dropout
と実行
"""

import ml, numpy as np, sys

(X_train, T_train), (X_test, T_test) = ml.load_data()

fnames = {
    'no_dropout' : 'pkl/mlp3d.pkl',
    'dropout'    : 'pkl/mlp3d_dropout_weight_decay.pkl'
}

which = fnames[sys.argv[1]]
net = ml.load(which)

net(X_train[:10000]) 
X = net[-2].z.copy() 
labels = T_train[:10000]
 
ev_train = ml.evaluate(ml.cluster.as_cluster(X, labels))
print(ev_train)
ml.save(ev_train, f'pkl/eval_mlp3d_{sys.argv[1]}_train')

net(X_test) 
X = net[-2].z.copy() 
labels = T_test

ev_test = ml.evaluate(ml.cluster.as_cluster(X, labels))
print(ev_test)
ml.save(ev_test, f'pkl/eval_mlp3d_{sys.argv[1]}_test')
