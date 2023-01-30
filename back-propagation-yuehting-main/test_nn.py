import math

import numpy as np
import pytest

import nn


@pytest.fixture(autouse=True)
def set_seed():
    np.random.seed(42)


@pytest.mark.timeout(2)
def test_predict():
    net = nn.SimpleNetwork(np.array([[.1, -.2],
                                     [-.3, .4],
                                     [.5, -.6]]),
                           np.array([[-.7, .8, -.9],
                                     [.10, -.11, .12]]))
    input_matrix = np.array([[13, -14, 15],
                             [-16, 17, -18]])
    predictions = net.predict(input_matrix)
    [[h11, h12],
     [h21, h22]] = [[s(13 * .1 + -14 * -.3 + 15 * .5),
                     s(13 * -.2 + -14 * .4 + 15 * -.6)],
                    [s(-16 * .1 + 17 * -.3 - 18 * .5),
                     s(-16 * -.2 + 17 * .4 + -18 * -.6)]]
    np.testing.assert_allclose(predictions, np.array(
        [[s(h11 * -.7 + h12 * .10),
          s(h11 * .8 + h12 * -.11),
          s(h11 * -.9 + h12 * .12)],
         [s(h21 * -.7 + h22 * .10),
          s(h21 * .8 + h22 * -.11),
          s(h21 * -.9 + h22 * .12)]]))

    binary_predictions = net.predict_zero_one(input_matrix)
    np.testing.assert_array_equal(binary_predictions, np.array([[0, 1, 0],
                                                                [1, 0, 1]]))


@pytest.mark.timeout(2)
def test_gradients():
    net = nn.SimpleNetwork(np.array([[.1, .3, .5],
                                     [-.5, -.3, -.1]]),
                           np.array([[.2, -.2],
                                     [.4, -.4],
                                     [.6, -.6]]))
    input_matrix = np.array([[0, 0],
                             [1, 0],
                             [0, 1],
                             [1, 1]])
    output_matrix = np.array([[1, 1],
                              [0, 1],
                              [1, 0],
                              [0, 0]])
    [[hi11, hi12, hi13],
     [hi21, hi22, hi23],
     [hi31, hi32, hi33],
     [hi41, hi42, hi43]] = hi = [
        [.1 * 0 + -.5 * 0, .3 * 0 + -.3 * 0, .5 * 0 + -.1 * 0],
        [.1 * 1 + -.5 * 0, .3 * 1 + -.3 * 0, .5 * 1 + -.1 * 0],
        [.1 * 0 + -.5 * 1, .3 * 0 + -.3 * 1, .5 * 0 + -.1 * 1],
        [.1 * 1 + -.5 * 1, .3 * 1 + -.3 * 1, .5 * 1 + -.1 * 1]]
    [[h11, h12, h13],
     [h21, h22, h23],
     [h31, h32, h33],
     [h41, h42, h43]] = [[s(x) for x in row] for row in hi]
    [[oi11, oi12],
     [oi21, oi22],
     [oi31, oi32],
     [oi41, oi42]] = oi = [
        [h11 * .2 + h12 * .4 + h13 * .6, h11 * -.2 + h12 * -.4 + h13 * -.6],
        [h21 * .2 + h22 * .4 + h23 * .6, h21 * -.2 + h22 * -.4 + h23 * -.6],
        [h31 * .2 + h32 * .4 + h33 * .6, h31 * -.2 + h32 * -.4 + h33 * -.6],
        [h41 * .2 + h42 * .4 + h43 * .6, h41 * -.2 + h42 * -.4 + h43 * -.6]]
    [[o11, o12],
     [o21, o22],
     [o31, o32],
     [o41, o42]] = [[s(x) for x in row] for row in oi]
    [[do11, do12],
     [do21, do22],
     [do31, do32],
     [do41, do42]] = [[(o11 - 1) * sg(oi11), (o12 - 1) * sg(oi12)],
                      [(o21 - 0) * sg(oi21), (o22 - 1) * sg(oi22)],
                      [(o31 - 1) * sg(oi31), (o32 - 0) * sg(oi32)],
                      [(o41 - 0) * sg(oi41), (o42 - 0) * sg(oi42)]]
    [[dh11, dh12, dh13],
     [dh21, dh22, dh23],
     [dh31, dh32, dh33],
     [dh41, dh42, dh43]] = [[(.2 * do11 + -.2 * do12) * sg(hi11),
                             (.4 * do11 + -.4 * do12) * sg(hi12),
                             (.6 * do11 + -.6 * do12) * sg(hi13)],
                            [(.2 * do21 + -.2 * do22) * sg(hi21),
                             (.4 * do21 + -.4 * do22) * sg(hi22),
                             (.6 * do21 + -.6 * do22) * sg(hi23)],
                            [(.2 * do31 + -.2 * do32) * sg(hi31),
                             (.4 * do31 + -.4 * do32) * sg(hi32),
                             (.6 * do31 + -.6 * do32) * sg(hi33)],
                            [(.2 * do41 + -.2 * do42) * sg(hi41),
                             (.4 * do41 + -.4 * do42) * sg(hi42),
                             (.6 * do41 + -.6 * do42) * sg(hi43)]]

    [input_to_hidden_gradient,
     hidden_to_output_gradient] = net.gradients(input_matrix, output_matrix)

    np.testing.assert_allclose(hidden_to_output_gradient, np.array(
        [[(do11 * h11 + do21 * h21 + do31 * h31 + do41 * h41) / 4,
          (do12 * h11 + do22 * h21 + do32 * h31 + do42 * h41) / 4],
         [(do11 * h12 + do21 * h22 + do31 * h32 + do41 * h42) / 4,
          (do12 * h12 + do22 * h22 + do32 * h32 + do42 * h42) / 4],
         [(do11 * h13 + do21 * h23 + do31 * h33 + do41 * h43) / 4,
          (do12 * h13 + do22 * h23 + do32 * h33 + do42 * h43) / 4]]))

    np.testing.assert_allclose(input_to_hidden_gradient, np.array(
        [[(0 * dh11 + 1 * dh21 + 0 * dh31 + 1 * dh41) / 4,
          (0 * dh12 + 1 * dh22 + 0 * dh32 + 1 * dh42) / 4,
          (0 * dh13 + 1 * dh23 + 0 * dh33 + 1 * dh43) / 4],
         [(0 * dh11 + 0 * dh21 + 1 * dh31 + 1 * dh41) / 4,
          (0 * dh12 + 0 * dh22 + 1 * dh32 + 1 * dh42) / 4,
          (0 * dh13 + 0 * dh23 + 1 * dh33 + 1 * dh43) / 4]]))


@pytest.mark.timeout(2)
def test_train_greater_than_half():
    inputs = np.random.uniform(size=(100, 1))
    outputs = (inputs > 0.5).astype(int)

    net = nn.SimpleNetwork.random(1, 5, 5, 1)
    assert len(net.gradients(inputs, outputs)) == 3
    net.train(inputs, outputs, iterations=1000, learning_rate=1)

    test_inputs = np.array([[0.0], [0.1], [0.2], [0.3], [0.4],
                            [0.6], [0.7], [0.8], [0.9], [1.0]])
    test_outputs = np.array([[0], [0], [0], [0], [0],
                             [1], [1], [1], [1], [1]])
    assert (net.predict_zero_one(test_inputs) == test_outputs).sum() >= 9


@pytest.mark.timeout(2)
def test_train_xor():
    inputs = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    outputs = np.array([[0], [1], [1], [0]])

    net = nn.SimpleNetwork.random(3, 3, 1)
    assert len(net.gradients(inputs, outputs)) == 2
    net.train(inputs, outputs, iterations=1000, learning_rate=0.5)

    assert np.all(net.predict_zero_one(inputs) == outputs)


@pytest.mark.timeout(2)
def test_train_learning_rate():
    inputs = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    outputs = np.array([[0], [1], [1], [1]])

    net = nn.SimpleNetwork.random(3, 3, 1)
    net.train(inputs, outputs, iterations=400, learning_rate=0.01)

    assert np.sometrue(net.predict_zero_one(inputs) != outputs)

    net = nn.SimpleNetwork.random(3, 3, 1)
    net.train(inputs, outputs, iterations=400, learning_rate=1)

    assert np.all(net.predict_zero_one(inputs) == outputs)


def s(x):
    return 1 / (1 + math.exp(-x))


def sg(x):
    return s(x) * (1 - s(x))
