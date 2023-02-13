import sys

import matplotlib.pylab as plt
import numpy as np

# zeroからdeepのload_mnist関数を参照
from common import load_mnist

# 1.パラメータの初期値(2層)
# w = (784, 10)
# b = (b1, b2)


# 2.必要な関数を定義
# softmax関数#
# TODO: ここ埋める
# 入力 x: ベクトル, どんな意味を持っている？
# 出力 y: ベクトル, y_0=...、y_1=...


# 2-1 sigmoid関数
# 目的：入力層の値を2層の入力の形にしたい
# 入力：入力x*重みw1+バイアスb1 の値
# 出力：値域 0~1
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# 2-2 softmax関数
# 目的：2層の入力を出力層の形(確率)にしたい
# 入力：2層の入力*重みw2+バイアスb2 の値
# 出力：値域 0~1
def softmax(x):
    y = np.zeros_like(x)
    # print("x.shape:", x.shape)
    # print("y.ndim:", y.ndim)

    # ミニバッチ数1の時
    if y.ndim == 1:
        max_x = np.max(x)  # max_x：入力xの最大値 OK
        # print("max_x:", max_x)
        x = x - max_x  # exp：入力xの最大値から引く
        # print("x:", x)
        exp = np.exp(x)
        # print("exp:", exp)
        sum_x = np.sum(exp)
        # print("sum_x:", sum_x)
        for i in range(x.shape[0]):
            y[i] = exp[i] / sum_x
            # print("softmax[{}].shape: {}, softmax: {}".format(i, y[i].shape, y[i]))
        # print("softmax:", y)

    # ミニバッチ数2以上の時
    else:
        for i in range(x.shape[0]):
            max_x = np.max(x[i])  # max_x：入力xの最大値 OK
            exp = np.exp(x[i] - max_x)  # exp：入力xの最大値から引く
            y[i] = exp / np.sum(exp)
            # print("max_x:", max_x)

    return y


# 2-3 predict(予測値)
# x(100, 784)=(A✖️B), w(784, 10)=(B✖️C), 欲しい形(100, 10)=(A✖️C)
# 100: バッチサイズ, 784: 特徴量(28*28)
# 入力 x:100のバッチサイズ、784の特徴量 (100x784(A✖️B)の行列)
# 入力 w:784✖️10行列 (重み)
# 出力 y：100✖️10
# 出力 y:100のバッチサイズ、10の各ラベルの予測の確率 y_0 = (0.1, 0.01, 0, 0.7, 0.02,....)
def predict(x, w):
    z = np.dot(x, w)
    # print("z:", z)
    # 各行の和が1になる
    # 非負になるはず

    y = softmax(z)

    # ミニバッチ数1の時
    if y.ndim == 1:
        new_y = y.reshape(-1, 1)
        # print("new_y:", new_y)
        assert round(np.sum(new_y)) == 1, "sum error"

    # ミニバッチ数2以上の時
    else:
        assert all([round(sum(a)) == 1 for a in y]), "sum error"
        assert all([a >= 0 for a in y.reshape(-1)]), "hihu error"

    return y


# 2-4 cross-entropy-error(yがニューラルネットの出力、tが教師データ)
# y(100, 10), t(100, 10)
# 100: バッチサイズ, 10:正解ラベルの種類
# 入力が満たすべき条件を列挙して、それを満たしているかを確認する
# 入力 y:100のバッチサイズ、10個の正解ラベルの種類 0<=y<=1 (100x10の行列) y = (0.1, 0.01, 0, 0.7, 0.02,....)
# y_(0~9)の総和は1
# 入力 t:100のバッチサイズ、10個の正解ラベルの種類 (100x10の行列) t0 = (0,0,0,1,0,0,0,...)
# tはone-hot
# 出力が満たすべき条件を列挙して、それを満たしているかを確認する
# 出力 スカラー  cross-entropyが1を超えるのは考えられる
# 正しい出力の例を用意して、対応する入力を入れた時にそれが出力されるかを確認する
# 例：ret = 0.8とか、ret = 1.3とか
# 0 < retは絶対
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    delta = 1e-7  # logの中をゼロにしたくないので微小な値を入れておく

    # xの範囲は1.0を超えることは容易に考えられる
    # log e^(-1)の時、-1を取るため
    # 正解データtがone-hotの場合(*今回 t = 0,0,0,1,0....)
    ret = -np.sum(t * np.log(y + delta)) / batch_size
    # print("cross_entropy_loss:", ret)

    # 正解データtが正解ラベル(ベクトル)の場合(t = 0~9までの数値)
    # ret = -np.sum(np.log(y[np.arange(batch_size), t] + delta)) / batch_size
    return ret


# 2-5 損失関数(xが入力画像、tが正解データ)
# x(100, 784), t(100, 10)
# 100: バッチサイズ, 784: 28*28画像サイズ ,10:正解ラベルの種類
# 入力が満たすべき条件を列挙して、それを満たしているかを確認する
# 入力 x:100のバッチサイズ、28*28画像サイズ (100x784の行列)
# 入力 t:100のバッチサイズ、10個の正解ラベルの種類 (100x10の行列)　t0 = (0,0,0,1,0,0,0,...)
# tはone-hot
# 出力が満たすべき条件を列挙して、それを満たしているかを確認する
# 出力 cross-entropy(スカラー)
# 出力：予測値yと正解tとの誤差を計算
# 正しい出力の例を用意して、対応する入力を入れた時にそれが出力されるかを確認する
def loss_function(x, t, w):
    y = predict(x, w)  # x:100*784
    # print("predict:", y)
    # y：100*10  (0.1, 0.5, 0.05, ・・・)
    # assert y.shape == (100, 10), 'predict shape error'

    return cross_entropy_error(y, t)  # スカラー


# 2-6 数値微分(fが微分したい関数、xが初期値, tが正解データ、wが重み(wで微分したい))
# 入力 f:微分したい関数(後々損失関数を微分したい)
# 入力 x:初期値(x0,x1,..など)
# 入力 t:100のバッチサイズ、10個の正解ラベルの種類 (100x10の行列)　t0 = (0,0,0,1,0,0,0,...)
# tはone-hot
# 入力 w:これで微分する(重みの更新更新)
# 内容：微分したい関数の結果を表示
# 出力 勾配 784✖️10行列
# ①cross-entropyをy(softmax)で微分
# ②y(softmax)をzで微分
# ③zをwで微分
def gradient(f, x, t, w):
    grad = np.zeros_like(w)  # wと同じ形状の配列を生成(要素が全て0)
    batch = x.shape[0]

    # ①と②の積
    # grad_12は微分①と微分②の結果の積
    grad_12 = np.zeros_like(t)
    n, k = grad_12.shape
    for i in range(n):
        for j in range(k):
            grad_12[i][j] = (1 / batch) * (f[i][j] - t[i][j])

    # ①、②、③のすべての積
    grad = np.dot(x.T, grad_12)

    return grad


# 2-7 accuracy(認識精度)
# 入力 x:入力データ t:正解データ w:重み
# 出力 スカラー(0~1の確率で表示)
def accuracy(x, t, w):
    y = predict(x, w)
    accuracy_y = np.argmax(y, axis=1)
    accuracy_t = np.argmax(t, axis=1)

    accuracy = np.sum(accuracy_y == accuracy_t) / float(x.shape[0])
    return accuracy
