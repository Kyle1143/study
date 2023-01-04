import sys

# デバッガーを使う
# set_trace()をおくと，その行で止まる
from pdb import set_trace

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
# 出力 y: ベクトル, y_0=...、y_1=...、
# 変数名は適当につけない方
# 正しいかどうか実際に実行してみてみよう
# 全体を実行するのではなく，この関数だけ切り出してみる
# 完成！

# 一つにしか対応していない -> 行列に対応していない


def softmax(x):
    y = np.zeros_like(x)
    # print("x.shape:", x.shape)
    # print("y.ndim:", y.ndim)

    # ミニバッチ数1の時
    if y.ndim == 1:
        max_x = np.max(x)  # max_x：入力xの最大値 OK
        # print("max_x:", max_x)
        new_x = x - max_x  # exp：入力xの最大値から引く
        # print("new_x:", new_x)
        exp = np.exp(new_x)
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


# 2-1predict(予測値)
# x(100, 784)=(A✖️B), w(784, 10)=(B✖️C), 欲しい形(100, 10)=(A✖️C)
# 100: バッチサイズ, 784: 特徴量(28*28)
# 入力 x:100のバッチサイズ、784の特徴量 (100x784(A✖️B)の行列)
# 入力 w:784✖️10行列 (重み)
# 出力 y:100のバッチサイズ、10の各ラベルの予測の確率 y_0 = (0.1, 0.01, 0, 0.7, 0.02,....)


def predict(x, w):
    seki = np.dot(x, w)
    # print("seki:", seki)
    # 各行の和が1になる
    # 非負になるはず

    y = softmax(seki)

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


# 2-2 cross-entropy-error(yがニューラルネットの出力、tが教師データ)
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


# 2-3損失関数(xが入力画像、tが正解データ)
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


# 2-4 数値微分(fが微分したい関数、xが初期値, tが正解データ、wが重み)
# 入力が満たすべき条件を列挙して、それを満たしているかを確認する
# 入力 f:微分したい関数(後々損失関数を微分したい)
# 入力 x:それについて微分する(x0,x1,..など)
# 入力 t:100のバッチサイズ、10個の正解ラベルの種類 (100x10の行列)　t0 = (0,0,0,1,0,0,0,...)
# tはone-hot
# 出力が満たすべき条件を列挙して、それを満たしているかを確認する
# 出力 勾配 784✖️10行列
# 正しい出力の例を用意して、対応する入力を入れた時にそれが出力されるかを確認する


def numerical_gradient(f, x, t, w):  # 引数：損失関数、初期値(x0やx1などの値)、正解データ、重み
    h = 1e-4
    grad = np.zeros_like(w)  # wと同じ形状の配列を生成(要素が全て0)
    # print("grad.shape():", grad.shape)
    tate, yoko = grad.shape  # grad.shape = (tate, yoko)

    for i in range(tate):
        for j in range(yoko):
            tmp_val = w[i][j]

            # f(x+h)の計算
            w[i][j] = tmp_val + h
            fxh1 = f(x, t, w)
            # print("fxh1:", fxh1)

            # f(x-h)の計算
            w[i][j] = tmp_val - h
            fxh2 = f(x, t, w)
            # print("fxh2:", fxh2)

            # f(w+h)-f(w-h)の計算と重み更新
            grad_fxh = fxh1 - fxh2
            # print("fxh1 - fxh2:", grad_fxh)
            grad[i][j] = grad_fxh / (2 * h)
            # print(
            #     "grad[{}].[{}].shape: {}, grad: {}".format(
            #         i, j, grad[i][j].shape, grad[i][j]
            #     )
            # )

    return grad


# 3.MNISTデータセットの読み込み
# load_mnist(normalize=True, flatten=True, one_hot_label=False):
# normalize : 画像のピクセル値を0.0~1.0に正規化する
# flatten : 画像を一次元配列に平にするかどうか
# one_hot_label :
#     one_hot_labelがTrueの場合、ラベルはone-hot配列として返す
#     one-hot配列とは、たとえば[0,0,1,0,0,0,0,0,0,0]のような配列
# 出力：(訓練画像, 訓練ラベル), (テスト画像, テストラベル)
# x_train.shape=(60000, 784), t_train.shape=(60000, 10)
# x_test.shape=(10000, 784), t_test.shape=(10000, 10)


(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
# print("x_train.shape: {}, x_train: {}".format(x_train.shape, x_train))
assert x_train.shape == (60000, 784), print("x_train Shape Size Error")
# print("t_train.shape: {}, t_train: {}".format(t_train.shape, t_train))
assert t_train.shape == (60000, 10), print("t_train Shape Size Error")
# print("x_test.shape: {}, x_test: {}".format(x_test.shape, x_test))
assert x_test.shape == (10000, 784), print("x_test Shape Size Error")
# print("t_test.shape: {}, t_test: {}".format(t_test.shape, t_test))
assert t_test.shape == (10000, 10), print("t_test Shape Size Error")


# # 4.ハイパラを作成(iter、train_size、test_size、学習率)
# iter = 100
# train_size = x_train.shape[0]
# lr = 0.01
# batch_size = 100

# # 5-1.ミニバッチを作成
# for i in range(iter):
#   a = np.random.choice(train_size, batch_size)
#   x_batch = x_train[a] #画像 100×784 (28*28=784)
#   t_batch = t_train[a] #画像のラベル 100×10

# # 5-2.勾配の計算
#   # y = loss_function(x_batch, t_batch) #損失値
#   grad = numerical_gradient(loss_function, x_batch)

# # 5-3.パラメータを更新
#   w -= grad * lr
