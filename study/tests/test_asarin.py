# asarin.pyで用いる関数をテスト
# テストしたい関数を呼び出す
import sys

sys.path.append(".")

import numpy as np
from asarin import (
    accuracy,
    cross_entropy_error,
    loss_function,
    numerical_gradient,
    predict,
    softmax,
)


def test_softmax():
    # Case1
    # 入力xを作る
    # 10個の数値がベクトルになっている x=[1,10,100,...,]みたいな
    x = np.array([10 ** (i + 1) for i in range(10)])
    print(f"x.shape:{x.shape}, x:{x}")
    # TODO: yの中身説明する
    # 出力y：10(size)次元のベクトル
    y = softmax(x)
    print("y.shape: {}, y: {}".format(y.shape, y))

    # Case2
    # 入力xを作る
    # 10個の数値がベクトルになっている x=[10,20,30,...,]みたいな
    x = np.array([10 * (i + 1) for i in range(10)])
    print(f"x.shape:{x.shape}, x:{x}")
    # TODO: yの中身説明する
    # 出力y：10(size)次元のベクトル
    y = softmax(x)
    print("y.shape: {}, y: {}".format(y.shape, y))

    # Case3
    # 入力xを作る
    # 個の数値がベクトルになっている x=[1,2,3,4,..]みたいな
    x = np.array([1, 2, 1, -1, 3, 2])
    print(f"x.shape:{x.shape}, x:{x}")
    # TODO: yの中身説明する
    # 出力y：ベクトル(要素6)
    y = softmax(x)
    print("y.shape: {}, y: {}".format(y.shape, y))


def test_predict():
    # Case1
    # 入力xを作る
    # ランダムに行列を生成
    x = np.random.rand(100, 784)
    # assert 条件式　を使うと確認が楽
    assert x.shape == (100, 784), print("Shape Size Error")
    # 入力wを作る
    w = np.random.rand(x.shape[1], 10)
    assert w.shape == (784, 10), print("w Shape Size Error")
    # print("x.shape: {}, x: {}".format(x.shape, x))
    y = predict(x, w)
    assert y.shape == (100, 10), print("Shape Error")
    # print("y.shape: {}, y: {}".format(y.shape, y))

    # Case2
    # 入力：ミニバッチ数が9, 特徴量次元が256を想定
    x = np.random.rand(9, 256)
    # 入力wを作る
    w = np.random.rand(x.shape[1], 10)
    assert w.shape == (256, 10), print("w Shape Size Error")
    # 出力：ミニバッチの数 x クラス数10を想定
    y = predict(x, w)
    assert x.shape == (9, 256)
    assert y.shape == (9, 10)
    assert len(x) == len(y)

    # Case3
    # 入力：ミニバッチ数が32, 特徴量次元が512を想定
    x = np.random.rand(32, 512)
    # 入力wを作る
    w = np.random.rand(x.shape[1], 3)
    assert w.shape == (512, 3), print("w Shape Size Error")
    # 出力：ミニバッチの数 x クラス数3を想定
    y = predict(x, w)
    assert x.shape == (32, 512)
    assert y.shape == (32, 3)
    assert len(x) == len(y)


def test_cross_entropy_error():
    # Case1
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
    a = np.random.randint(10, size=(100))
    # print("a.shape: {}, a: {}".format(a.shape, a))
    assert a.shape == (100,), print("shape error02")
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

    # 次元数が1の時とそれ以外で場合分け
    if pre.ndim == 1:
        t = t.reshape(1, t.size)
        pre = pre.reshape(1, pre.size)

    # # Case2
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


def test_loss_function():
    # Case1：想定するshapeでの計算ができるかチェック
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
    # 入力 w
    w = np.random.rand(784, 10)
    assert w.shape == (784, 10), print("w Shape Size Error")
    # loss関数に入力してチェック
    L = loss_function(x, t, w)
    print("L.shape: {}, loss_function: {}".format(L.shape, L))
    assert isinstance(L, float), "Shape Error"

    # Case2：ちゃんと計算結果が間違っていないかチェック
    # 手計算用
    # x.shape=(2,2),t.shape=(2,3),w.shape=(2,3),true_grad.shape=(2,3)
    x1 = [[0.5, 0.8], [0.7, 0.9]]
    t1 = [[1, 0, 0], [0, 1, 0]]
    x = np.array(x1)
    assert x.shape == (2, 2), "x shape error"
    t = np.array(t1)
    assert t.shape == (2, 3), "t shape error"
    w1 = [[1.2, 1.5, 1.0], [2.0, 1.7, 2.0]]
    w = np.array(w1)
    assert w.shape == (2, 3), "w shape error"
    # loss関数に入力してチェック
    L = loss_function(x, t, w)
    print("L.shape: {}, loss_function: {}".format(L.shape, L))
    assert isinstance(L, float), "Shape Error"


def test_numerical_gradient():
    x = np.array([0.6, 0.9])
    t = np.array([0, 0, 1])
    # t = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0]])

    # case1
    w = np.array(
        [[-0.14161869, 0.90499692, 0.49966265], [1.73311584, -1.08183988, 1.56606182]]
    )
    # 手計算結果をここで記載
    true_grad = np.array(
        [[0.24866079, 0.0369882, -0.28564899], [0.37299119, 0.0554823, -0.42847348]]
    )
    # TODO: 実装した勾配の式を入れる
    y = numerical_gradient(loss_function, x, t, w)
    print("numerical_gradient.shape: {}, numerical_gradient: {}".format(y.shape, y))
    assert (abs(true_grad - y) < 1e-3).all()
    assert w.shape == y.shape

    # case2
    w = np.array(
        [[-1.11403581, -0.14819338, 2.90998236], [0.37014269, -1.41260429, -0.05247394]]
    )
    true_grad = np.array(
        [[0.06663823, 0.02391035, -0.09054858], [0.09995734, 0.03586553, -0.13582287]]
    )
    y = numerical_gradient(loss_function, x, t, w)
    print("numerical_gradient.shape: {}, numerical_gradient: {}".format(y.shape, y))
    assert (abs(true_grad - y) < 1e-3).all()
    assert w.shape == y.shape

    # case3
    # x.shape=(2,2),t.shape=(2,3),w.shape=(2,3),true_grad.shape=(2,3)
    x1 = [[0.5, 0.8], [0.7, 0.9]]
    t1 = [[1, 0, 0], [0, 1, 0]]
    x = np.array(x1)
    assert x.shape == (2, 2), "x shape error"
    t = np.array(t1)
    assert t.shape == (2, 3), "t shape error"
    w1 = [[1.2, 1.5, 1.0], [2.0, 1.7, 2.0]]
    w = np.array(w1)
    assert w.shape == (2, 3), "w shape error"

    # 手計算結果をここで記載
    true_grad = [
        [-0.0368, -0.1516, 0.1884],
        [-0.0980, -0.1695, 0.2675],
    ]
    true_grad = np.array(true_grad)
    assert true_grad.shape == (2, 3), "true_grad shape error"
    # numerical_gradient
    y = numerical_gradient(loss_function, x, t, w)
    assert y.shape == (2, 3), print("y Shape Size Error")
    print("numerical_gradient.shape: {}, numerical_gradient: {}".format(y.shape, y))

    # 手計算結果とcodeの出力結果を確認(< 1e-4)
    assert (abs(true_grad - y) < 1e-4).all(), print("true_grad - y error")
    # print(
    #     "true_grad - numerical_gradient: {}, true_grad - numerical_gradient: {}".format(
    #         (true_grad - y).shape, true_grad - y
    #     )
    # )
    assert w.shape == y.shape, print("w.shape == y.shape error")

    # Case4：本番のshape
    # 入力xを作る
    # ランダムに行列を生成
    x = np.random.rand(100, 784)
    assert x.shape == (100, 784), print("x Shape Size Error")
    # 本番のshape：入力tを作る
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
    # 入力 w
    w = np.random.rand(784, 10)
    assert w.shape == (784, 10), print("w Shape Size Error")
    # numerical_gradient(関数(loss), 微分したい変数(x))
    y = numerical_gradient(loss_function, x, t, w)
    assert y.shape == (784, 10), print("y Shape Size Error")
    print("y.shape: {}, numerical_gradient: {}".format(y.shape, y))


def test_accuracy():
    # Case1：想定するshapeでの計算ができるかチェック
    # 入力xを作る
    # ランダムに行列を生成
    # x = np.random.rand(100, 784)
    # assert x.shape == (100, 784), print("Shape Size Error")
    # # 入力tを作る
    # a = np.random.randint(10, size=(100))
    # # print("a.shape: {}, a: {}".format(a.shape, a))
    # assert a.shape == (100,), print("shape error02")
    # a_one_hot = np.identity(10)[a]
    # assert a_one_hot.shape == (100, 10), print("shape error03")
    # t = a_one_hot
    # # print("t.shape: {}, t: {}".format(t.shape, t))
    # assert t.shape == (100, 10), print("shape error04")
    # # one-hotの確認assert
    # assert all([sum(b) == 1 for b in t]), print("one-hot error05")
    # # 入力 w
    # w = np.random.rand(784, 10)
    # assert w.shape == (784, 10), print("w Shape Size Error")
    # # axxuracy関数に入力してチェック
    # A = accuracy(x, t, w)
    # print("A.shape: {}, accuracy: {}".format(A.shape, A))

    # case2
    # x.shape=(2,2),t.shape=(2,3),w.shape=(2,3),true_grad.shape=(2,3)
    x1 = [[0.5, 0.8], [0.7, 0.9]]
    t1 = [[1, 0, 0], [0, 1, 0]]
    x = np.array(x1)
    assert x.shape == (2, 2), "x shape error"
    t = np.array(t1)
    assert t.shape == (2, 3), "t shape error"
    w1 = [[1.2, 1.5, 1.0], [2.0, 1.7, 2.0]]
    w = np.array(w1)
    assert w.shape == (2, 3), "w shape error"
    # axxuracy関数に入力してチェック
    A = accuracy(x, t, w)
    print("A.shape: {}, accuracy: {}".format(A.shape, A))


if __name__ == "__main__":
    pass
