# テストしたい関数を呼び出す
import sys

sys.path.append(".")

import numpy as np
from asarin import (
    cross_entropy_error,
    loss_function,
    numerical_gradient,
    predict,
    softmax,
)


def test_softmax():
    # 入力xを作る
    # 10個の数値がベクトルになっている x=[1,10,100,...,]みたいな
    x = np.array([10 ** (i + 1) for i in range(10)])
    print(f"x.shape:{x.shape}, x:{x}")
    # TODO: yの中身説明する
    # 出力y：10(size)次元のベクトル
    y = softmax(x)
    print("y.shape: {}, y: {}".format(y.shape, y))

    # 入力xを作る
    # 10個の数値がベクトルになっている x=[10,20,30,...,]みたいな
    x = np.array([10 * (i + 1) for i in range(10)])
    print(f"x.shape:{x.shape}, x:{x}")
    # TODO: yの中身説明する
    # 出力y：10(size)次元のベクトル
    y = softmax(x)
    print("y.shape: {}, y: {}".format(y.shape, y))

    # 入力xを作る
    # 個の数値がベクトルになっている x=[1,2,3,4,..]みたいな
    x = np.array([1, 2, 1, -1, 3, 2])
    print(f"x.shape:{x.shape}, x:{x}")
    # TODO: yの中身説明する
    # 出力y：ベクトル(要素6)
    y = softmax(x)
    print("y.shape: {}, y: {}".format(y.shape, y))


def test_predict():
    # 入力xを作る
    # x = (100, 784)
    # x = [[ i for i in range(100)] for _ in range(784)]
    # ランダムに行列を生成
    x = np.random.rand(100, 784)

    # assert 条件式　を使うと確認が楽
    assert x.shape == (100, 784), print("Shape Size Error")

    # print("x.shape: {}, x: {}".format(x.shape, x))
    y = predict(x)
    assert y.shape == (100, 10), print("Shape Error")
    # print("y.shape: {}, y: {}".format(y.shape, y))
    # print("predict:", y)


def test_cross_entropy_error():
    # 予測値 0~1の範囲の実数 100✖️10の行列
    y = np.random.rand(100, 10)
    assert y.shape == (100, 10), print("shape error01")
    # print("y.shape: {}, y: {}".format(y.shape, y))

    # yの行の総和の逆数でその行の成分を割りたい -> 各行の確率の総和が1になる
    # ex:各行の総和
    # pre：確率に正規化
    ex = np.sum(y, axis=0)
    pre = y / ex

    # 正解ラベル 正解のラベルのインデックスに1、それ以外０(one-hot)　100✖️10行列
    # aは100の正解ラベル

    # 書き方が間違い?
    # a = np.random.randint(0, 9) for i in range(100)

    a = np.random.randint(10, size=(100))

    # a_list = []
    # for k in range(100):
    #     x = np.random.randint(0,9)
    #     a_list.append(x)
    # aa = np.array(a_list)
    # a = aa.astype(int)

    # print("a.shape: {}, a: {}".format(a.shape, a))
    assert a.shape == (100,), print("shape error02")
    # a.ndim = 1
    # print(type(a))
    # <class 'numpy.ndarray'>
    a_one_hot = np.identity(10)[a]
    assert a_one_hot.shape == (100, 10), print("shape error03")
    t = a_one_hot
    # print("t.shape: {}, t: {}".format(t.shape, t))
    assert t.shape == (100, 10), print("shape error04")
    # print(type(t))
    # <class 'numpy.ndarray'>

    # one-hotの確認assert
    assert all([sum(b) == 1 for b in t]), print("one-hot error05")

    # # print(y.ndim)
    # # ndim = 2

    if pre.ndim == 1:
        t = t.reshape(1, t.size)
        pre = pre.reshape(1, pre.size)

    # # ここだけ手計算用
    # #-0.69314718055995 + -0.35667494393873 + -0.22314355131421
    # # 0.42431667
    # y1 = [[0.5, 0.4, 0.1],[0.7, 0.2, 0.1],[0.1, 0.8, 0.1]]
    # t1 = [[1, 0, 0],[1, 0, 0],[0, 1, 0]]
    # y = np.array(y1)
    # t = np.array(t1)

    c = cross_entropy_error(y, t)
    assert c.shape == (), "Shape Error"
    print("c.shape: {}, cross-entropy: {}".format(c.shape, c))

    # #一旦ここまでテスト
    # return


def test_loss_function():
    # 入力xを作る
    # ランダムに行列を生成
    x = np.random.rand(100, 784)
    assert x.shape == (100, 784), print("Shape Size Error")

    # 入力tを作る
    a = np.random.randint(10, size=(100))
    # print("a.shape: {}, a: {}".format(a.shape, a))
    assert a.shape == (100,), print("shape error02")
    a_one_hot = np.identity(10)[a]
    assert a_one_hot.shape == (100, 10), print("shape error03")
    t = a_one_hot
    # print("t.shape: {}, t: {}".format(t.shape, t))
    assert t.shape == (100, 10), print("shape error04")
    # one-hotの確認assert
    assert all([sum(b) == 1 for b in t]), print("one-hot error05")

    # ここだけ手計算用
    # x1 (3✖️9行列)
    x1 = [
        [0.1, 0.8, 0.1, 0.1, 0.8, 0.1, 0.1, 0.8, 0.1],
        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.1],
        [0.1, 0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1],
    ]
    t1 = [[1, 0, 0], [1, 0, 0], [0, 1, 0]]
    x = np.array(x1)
    assert x.shape == (3, 9), "x shape error"
    t = np.array(t1)
    assert t.shape == (3, 3), "t shape error"

    L = loss_function(x, t)
    assert isinstance(L, float), "Shape Error"


def test_numerical_gradient():
    # # 入力xを作る
    # # ランダムに行列を生成
    # x = np.random.rand(100, 784)
    # assert x.shape == (100,784), print('x Shape Size Error')

    # # 入力tを作る
    # a = np.random.randint(10,size=(100))
    # # print("a.shape: {}, a: {}".format(a.shape, a))
    # assert a.shape == (100, ), print("shape error02")
    # a_one_hot = np.identity(10)[a]
    # assert a_one_hot.shape == (100, 10), print("shape error03")
    # t = a_one_hot
    # # print("t.shape: {}, t: {}".format(t.shape, t))
    # assert t.shape == (100, 10), print("shape error04")
    # # one-hotの確認assert
    # assert all([sum(b)==1 for b in t]), print('one-hot error05')

    # # numerical_gradient(関数(loss), 微分したい変数(x))
    # y = numerical_gradient(loss_function, x, t)
    # assert y.shape == (100,784), print('y Shape Size Error')
    # print("y.shape: {}, numerical_gradient: {}".format(y.shape, y))

    # ここだけ手計算用
    # x1 (3✖️9行列)
    x1 = [
        [0.1, 0.8, 0.1, 0.1, 0.8, 0.1, 0.1, 0.8, 0.1],
        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.1],
        [0.1, 0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1],
    ]
    t1 = [[1, 0, 0], [1, 0, 0], [0, 1, 0]]
    x = np.array(x1)
    assert x.shape == (3, 9), "x shape error"
    t = np.array(t1)
    assert t.shape == (3, 3), "t shape error"

    # numerical_gradient(関数(loss), 微分したい変数(x))
    y = numerical_gradient(loss_function, x, t)
    assert y.shape == (3, 9), print("y Shape Size Error")
    print("y.shape: {}, numerical_gradient: {}".format(y.shape, y))


if __name__ == "__main__":
    pass
