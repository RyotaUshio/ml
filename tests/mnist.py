import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import Size, Divider
import ml


def hidden_slideshow(net, dt=0.5):
    weights = net[1].W.T
    fig, ax = plt.subplots()
    plt.ion()
    for weight in weights:
        ml.imshow(weight, ax=ax)
        plt.pause(dt)
        
    

def hidden_image_and_bars(net, figsize=(8-1/4, 11-3/4)):
    size = figsize
    rect = (0.15, 0.15, 0.79, 0.79)
    pad_width = size[0]*.05/9.0 * rect[2]
    pad_height = pad_width*5
    im_height = im_width = (size[0] * rect[2] - pad_width*9)/10
    bar_height = (size[1]*rect[2] - pad_height*10 - im_height)/10
    bar_width = size[0] * rect[2]
    
    fig = plt.figure(figsize=size)

    # fixed size in inch
    horiz = ([Size.Fixed(im_width), Size.Fixed(pad_width)] * 10)[:-1]
    vert =  [Size.Fixed(im_height)] + [Size.Fixed(pad_height), Size.Fixed(bar_height)]*10
    vert[1] = Size.Fixed(pad_height*2)

    # divide the axes rectangle into grid whose size is specified by horiz * vert
    divider = Divider(fig, rect, horiz, vert, aspect=False)
    plt.ion()

    reps = rep()
    w_min, w_max = np.inf, -np.inf
    for last in range(10):
        # last: 出力層の各ニューロンの番号
        ax = fig.add_axes(rect, label=f"bar{last}")
        ax.set_axes_locator(divider.new_locator(nx=0, nx1=-1, ny=20-2*last))
        w = net[-1].W[:, last]
        if w.max() > w_max:
            w_max = w.max()
        if w.min() < w_min:
            w_min = w.min()
        ax.bar(x=range(10), height=w)
        ax.plot(np.linspace(-0.5, 9.5, 10), np.full(10, 0), linestyle='--', color='k', linewidth=.8)
        ax.xaxis.set_visible(False)

    w_range = max(abs(w_max), abs(w_min))
    for last, ax in zip(range(10), fig.axes):
        ax.set(xlim=(-0.5, 9.5), ylim=(-w_range, w_range), ylabel=f'{last}')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.xaxis.set_visible(True)

    for hidden in range(10):
        # hidden: 中間層の各ニューロンの番号
        ax = fig.add_axes(rect, label=f"image{hidden}")
        ax.set_axes_locator(divider.new_locator(nx=hidden*2, ny=0))
        w = normalize(net[1].W[:, hidden])
        imshow(w, ax=ax)

    fontsize = 'large'
    fig.text(rect[0]+0.5*rect[2], 0.12, 'Hidden Units', ha='center', va='center', fontsize=fontsize)
    fig.text(0.06, rect[1]+0.5*rect[3], 'Output Units', ha='center', va='center', rotation='vertical', fontsize=fontsize)
    
def rep():
    if cur_data != keras.datasets.mnist:
        raise Exception(f"expected cur_data == keras.datasets.mnist, but now cur_data == {cur_data}")
    reps = []
    for i, j in zip(range(10), [0, 0, 2, 0, 16, 19, 5, 0, 2, 2]):
        reps.append(X_train[vec2num(T_train) == i][j])
    return np.array(reps)
