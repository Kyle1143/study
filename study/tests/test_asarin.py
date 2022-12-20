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
    # print("predict:", y)

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

    # 入力 w
    w = np.random.rand(784, 10)
    assert w.shape == (784, 10), print("w Shape Size Error")

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

    # 手計算結果をここで記載
    true_grad = None
    # true_grad = [[1.2, 1.5, 1.0], [2.0, 1.7, 2.0]]
    # true_grad = np.array(true_grad)
    # assert true_grad.shape == (2, 3), "true_grad shape error"

    L = loss_function(x, t, w)
    print("L.shape: {}, loss_function: {}".format(L.shape, L))
    assert isinstance(L, float), "Shape Error"


def test_numerical_gradient():
    # 入力xを作る
    # ランダムに行列を生成
    x = np.random.rand(100, 784)
    assert x.shape == (100, 784), print("x Shape Size Error")

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

    # numerical_gradient(関数(loss), 微分したい変数(x))
    y = numerical_gradient(loss_function, x, t, w)
    assert y.shape == (784, 10), print("y Shape Size Error")
    print("y.shape: {}, numerical_gradient: {}".format(y.shape, y))

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

    # 手計算結果をここで記載
    # true_grad = None
    true_grad = [
        [-2.12988409, -2.129876795, -2.12995633],
        [-2.129884091, -2.129884091, -2.129884091],
    ]
    true_grad = np.array(true_grad)
    assert true_grad.shape == (2, 3), "true_grad shape error"

    # numerical_gradient
    y = numerical_gradient(loss_function, x, t, w)
    assert y.shape == (2, 3), print("y Shape Size Error")
    print("y.shape: {}, numerical_gradient: {}".format(y.shape, y))

    # 手計算結果とcodeの出力結果を確認
    assert ((true_grad - y) < 1e-3).all()
    assert w.shape == y.shape

    # 行列積 np.dot(x,w)
    # 2.2	2.11	2.1
    # 2.64	2.58	2.5

    # softmax
    # ①
    # 0.35475339738891
    # 0.33502200426645

    # 0.35477628806381
    # 0.33500531986078

    # ②
    # 0.35477059387453
    # 0.35476426751456

    # 0.33499806776239
    # 0.33502925689354

    # ③
    # 0.35477053664791
    # 0.35470789699171

    # 0.33502091441377
    # 0.33500640968829

    # 4
    # 0.3547465303161
    # 0.35478315539594

    # 0.33502438769796
    # 0.33500293631923

    # 5
    # 0.35477404456478
    # 0.35475564046292

    # 0.33499361228984
    # 0.33503371260132

    # 6
    # 0.35477395300178
    # 0.3547557320238

    # 0.33502298645417
    # 0.33500433752141

    # exp
    # ①
    # -1.036332386134 + -1.0935590649367
    # -1.0362678626183 + -1.0936088671017

    # #②
    # -1.0362839128312 + -1.036301745246
    # -1.0936305150472 + -1.0935374169599

    # #③
    # -1.0362840741372 + -1.0364606535888
    # -1.0935623180199 + -1.0936056139422

    # #4
    # -1.0363517436351 + -1.0362485060099
    # -1.0935519507084 +-1.0936159820635

    # #5
    # -1.0362741863389 + -1.0363260632434
    # -1.0936438151304 + -1.0935241175896

    # #6
    # -1.0362744444271 + -1.0363258051478
    # -1.09355613323 + -1.0936117994158

    # cross_entropy
    # -2.12988409
    # -2.129876795
    # -2.12995633

    # -2.129884091
    # -2.129884091
    # -2.129884091


if __name__ == "__main__":
    pass
