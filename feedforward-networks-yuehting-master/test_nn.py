import os
from typing import List

import h5py
import numpy as np
import pytest
import tensorflow

import nn


@pytest.fixture(autouse=True)
def set_seeds():
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    tensorflow.random.set_seed(42)
    tensorflow.config.threading.set_intra_op_parallelism_threads(1)
    tensorflow.config.threading.set_inter_op_parallelism_threads(1)


def test_deep_vs_wide(capsys):
    # The data for this test was obtained as follows:
    #
    # import pandas as pd
    # import numpy as np
    # import h5py
    # url = "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
    # df = pd.read_csv(url, header=None, sep="\s+", na_values="?", names=[
    #     "mpg", "cylinders", "displacement", "horsepower", "weight",
    #     "acceleration", "model year", "origin", "carname"])
    # df = df.dropna().drop("carname", axis=1)
    # input_df = df.drop("mpg", axis=1)
    # output_df = df[["mpg"]]
    # mask = np.random.rand(len(df)) < 0.8
    # train_input = input_df[mask].as_matrix()
    # train_output = output_df[mask].as_matrix()
    # test_input = input_df[~mask].as_matrix()
    # test_output = output_df[~mask].as_matrix()
    # with h5py.File('data/auto-mpg.hdf5', 'w') as f:
    #     train = f.create_group("train")
    #     train.create_dataset("input", compression="gzip", data=train_input)
    #     train.create_dataset("output", compression="gzip", data=train_output)
    #     test = f.create_group("test")
    #     test.create_dataset("input", compression="gzip", data=test_input)
    #     test.create_dataset("output", compression="gzip", data=test_output)

    train_in, train_out, test_in, test_out = load_hdf5("data/auto-mpg.hdf5")

    deep, wide = nn.create_auto_mpg_deep_and_wide_networks(
        train_in.shape[-1], train_out.shape[-1])

    # check that the deep neural network is indeed deeper
    assert len(deep.layers) > len(wide.layers)

    # check that the 2 networks have (nearly) the same number of parameters
    params1 = deep.count_params()
    params2 = wide.count_params()
    assert abs(params1 - params2) / (params1 + params2) < 0.05

    # check that the 2 networks have the same compile parameters
    assert_compile_parameters_equal(deep, wide)

    # check that the 2 networks have the same activation functions
    assert set(hidden_activations(deep)) == set(hidden_activations(wide))

    # check that output type and loss are appropriate for regression
    assert all("mean" in loss_name(model) for model in [deep, wide])
    assert loss_name(deep) == loss_name(wide)
    assert output_activation(deep) == output_activation(wide) == \
        tensorflow.keras.activations.linear

    # train both networks
    deep.fit(train_in, train_out, verbose=0, epochs=100)
    wide.fit(train_in, train_out, verbose=0, epochs=100)

    # check that error level is acceptable
    mean_predict = np.full(shape=test_out.shape, fill_value=np.mean(train_out))
    [baseline_rmse] = root_mean_squared_error(mean_predict, test_out)
    [deep_rmse] = root_mean_squared_error(deep.predict(test_in), test_out)
    [wide_rmse] = root_mean_squared_error(wide.predict(test_in), test_out)
    with capsys.disabled():
        rmse_format = "{1:.1f} RMSE for {0} on Auto MPG".format
        print()
        print(rmse_format("baseline", baseline_rmse))
        print(rmse_format("deep", deep_rmse))
        print(rmse_format("wide", wide_rmse))

    assert deep_rmse < baseline_rmse
    assert wide_rmse < baseline_rmse


def test_relu_vs_tanh(capsys):
    # The data for this test was obtained as follows:
    #
    # import os
    # import numpy as np
    # import h5py
    # os.system("git clone https://github.com/dhruvramani/Multilabel-Classification-Datasets.git")
    # file_format = 'Multilabel-Classification-Datasets/delicious/delicious-{0}.pkl'.format
    # train_input = np.load(file_format('train-features'))
    # train_output = np.load(file_format('train-labels'))
    # test_input = np.load(file_format('test-features'))
    # test_output = np.load(file_format('test-labels'))
    # with h5py.File('data/delicious.hdf5', 'w') as f:
    #     train = f.create_group("train")
    #     train.create_dataset("input", compression="gzip", data=train_input)
    #     train.create_dataset("output", compression="gzip", data=train_output)
    #     test = f.create_group("test")
    #     test.create_dataset("input", compression="gzip", data=test_input)
    #     test.create_dataset("output", compression="gzip", data=test_output)

    train_in, train_out, test_in, test_out = load_hdf5("data/delicious.hdf5")

    # keep only every 10th training example
    train_out = train_out[::10, :]
    train_in = train_in[::10, :]
    # keep only tags that occur at least 400 times
    (tags,) = np.nonzero(np.sum(train_out, axis=0) >= 400)
    train_out = train_out[:, tags]
    test_out = test_out[:, tags]

    relu, tanh = nn.create_delicious_relu_vs_tanh_networks(
        train_in.shape[-1], train_out.shape[-1])

    # check that models are identical other than the activation functions
    assert len(relu.layers) == len(tanh.layers)
    for relu_layer, tanh_layer in zip(relu.layers, tanh.layers):
        assert relu_layer.__class__ == tanh_layer.__class__
        assert getattr(relu_layer, "units", None) == \
               getattr(tanh_layer, "units", None)

    # check that relu layers are all relu
    relu_activations = hidden_activations(relu)
    assert relu_activations
    assert all(a == tensorflow.keras.activations.relu for a in relu_activations)

    # check that tanh layers are all tanh
    tanh_activations = hidden_activations(tanh)
    assert tanh_activations
    assert all(a == tensorflow.keras.activations.tanh for a in tanh_activations)

    # check that the 2 networks have the same number of parameters
    assert relu.count_params() == tanh.count_params()

    # check that the 2 networks have the same compile parameters
    assert_compile_parameters_equal(relu, tanh)

    # check that output type and loss are appropriate for multi-label
    assert all(any(x in loss_name(model) for x in {"crossentropy", "hinge"})
               and "categorical" not in loss_name(model)
               for model in [relu, tanh])
    assert loss_name(relu) == loss_name(tanh)
    assert output_activation(relu) == output_activation(tanh) == \
        tensorflow.keras.activations.sigmoid

    # train both networks
    relu.fit(train_in, train_out, verbose=0, epochs=10)
    tanh.fit(train_in, train_out, verbose=0, epochs=10)

    # check that error levels are acceptable
    relu_accuracy = binary_accuracy(relu.predict(test_in), test_out)
    tanh_accuracy = binary_accuracy(tanh.predict(test_in), test_out)
    all0_accuracy = np.sum(test_out == 0) / test_out.size
    with capsys.disabled():
        accuracy_format = "{1:.1%} accuracy for {0} on del.icio.us".format
        print()
        print(accuracy_format("baseline", all0_accuracy))
        print(accuracy_format("relu", relu_accuracy))
        print(accuracy_format("tanh", tanh_accuracy))
    assert relu_accuracy > all0_accuracy
    assert tanh_accuracy > all0_accuracy


def test_dropout(capsys):
    # The data for this test was obtained as follows:
    #
    # import io
    # import zipfile
    # import urllib.request
    # import numpy as np
    # import h5py
    # from keras.utils import to_categorical
    # url = "http://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
    # with zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(url).read())) as zip:
    #     train_input = np.loadtxt(
    #         zip.extract("UCI HAR Dataset/train/X_train.txt"))
    #     train_output = to_categorical(np.loadtxt(
    #         zip.extract("UCI HAR Dataset/train/y_train.txt")))
    #     test_input = np.loadtxt(
    #         zip.extract("UCI HAR Dataset/test/X_test.txt"))
    #     test_output = to_categorical(np.loadtxt(
    #         zip.extract("UCI HAR Dataset/test/y_test.txt")))
    #
    # with h5py.File('data/uci-har.hdf5', 'w') as f:
    #     train = f.create_group("train")
    #     train.create_dataset("input", compression="gzip", data=train_input, dtype=np.dtype("f2"))
    #     train.create_dataset("output", compression="gzip", data=train_output, dtype=np.dtype("i1"))
    #     test = f.create_group("test")
    #     test.create_dataset("input", compression="gzip", data=test_input, dtype=np.dtype("f2"))
    #     test.create_dataset("output", compression="gzip", data=test_output, dtype=np.dtype("i1"))

    train_in, train_out, test_in, test_out = load_hdf5("data/uci-har.hdf5")

    # keep only every 10th training example
    train_out = train_out[::10, :]
    train_in = train_in[::10, :]

    drop, no_drop = nn.create_activity_dropout_and_nodropout_networks(
        train_in.shape[-1], train_out.shape[-1])

    # check that the dropout network has Dropout and the other doesn't
    assert any(isinstance(layer, tensorflow.keras.layers.Dropout)
               for layer in drop.layers)
    assert all(not isinstance(layer, tensorflow.keras.layers.Dropout)
               for layer in no_drop.layers)

    # check that the 2 networks have the same number of parameters
    assert drop.count_params() == no_drop.count_params()

    # check that the two networks are identical other than dropout
    dropped_dropout = [l for l in drop.layers
                       if not isinstance(l, tensorflow.keras.layers.Dropout)]
    assert_layers_equal(dropped_dropout, no_drop.layers)

    # check that the 2 networks have the same compile parameters
    assert_compile_parameters_equal(drop, no_drop)

    # check that output type and loss are appropriate for multi-class
    assert all("categorical" in loss_name(model)
               for model in [drop, no_drop])
    assert loss_name(drop) == loss_name(no_drop)
    assert output_activation(drop) == output_activation(no_drop) == \
        tensorflow.keras.activations.softmax

    # train both networks
    drop.fit(train_in, train_out, verbose=0, epochs=10)
    no_drop.fit(train_in, train_out, verbose=0, epochs=10)

    # check that accuracy level is acceptable
    baseline_prediction = np.zeros_like(test_out)
    baseline_prediction[:, np.argmax(np.sum(train_out, axis=0), axis=0)] = 1
    baseline_accuracy = multi_class_accuracy(baseline_prediction, test_out)
    dropout_accuracy = multi_class_accuracy(drop.predict(test_in), test_out)
    no_dropout_accuracy = multi_class_accuracy(
        no_drop.predict(test_in), test_out)
    with capsys.disabled():
        accuracy_format = "{1:.1%} accuracy for {0} on UCI-HAR".format
        print()
        print(accuracy_format("baseline", baseline_accuracy))
        print(accuracy_format("dropout", dropout_accuracy))
        print(accuracy_format("no dropout", no_dropout_accuracy))
    assert dropout_accuracy >= 0.75
    assert no_dropout_accuracy >= 0.75


def test_early_stopping(capsys):
    # The data for this test was obtained as follows:
    #
    # import pandas as pd
    # import numpy as np
    # import h5py
    # url = "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    # df = pd.read_csv(url, header=None, sep=", ", na_values="?",
    #                  engine="python", names="""age workclass fnlwgt education
    #                  education-num marital-status occupation relationship race
    #                  sex capital-gain capital-loss hours-per-week
    #                  native-country income""".split())
    # df = df.dropna()
    # df = pd.get_dummies(df)
    # df = df.drop("income_>50K", axis=1)
    # input_df = df.drop("income_<=50K", axis=1)
    # output_df = df[["income_<=50K"]]
    # mask = np.random.rand(len(df)) < 0.8
    # train_input = input_df[mask].as_matrix()
    # train_output = output_df[mask].as_matrix()
    # test_input = input_df[~mask].as_matrix()
    # test_output = output_df[~mask].as_matrix()
    # with h5py.File('data/income.hdf5', 'w') as f:
    #     train = f.create_group("train")
    #     train.create_dataset("input", compression="gzip", data=train_input)
    #     train.create_dataset("output", compression="gzip", data=train_output)
    #     test = f.create_group("test")
    #     test.create_dataset("input", compression="gzip", data=test_input)
    #     test.create_dataset("output", compression="gzip", data=test_output)

    train_in, train_out, test_in, test_out = load_hdf5("data/income.hdf5")

    # keep only every 10th training example
    train_out = train_out[::10, :]
    train_in = train_in[::10, :]

    early, early_fit_kwargs, late, late_fit_kwargs = \
        nn.create_income_earlystopping_and_noearlystopping_networks(
            train_in.shape[-1], train_out.shape[-1])

    # check that the two networks have the same number of parameters
    assert early.count_params() == late.count_params()

    # check that the two networks have identical layers
    assert_layers_equal(early.layers, late.layers)

    # check that the 2 networks have the same compile parameters
    assert_compile_parameters_equal(early, late)

    # check that output type and loss are appropriate for binary classification
    assert all(any(x in loss_name(model) for x in {"crossentropy", "hinge"})
               and "categorical" not in loss_name(model)
               for model in [early, late])
    assert loss_name(early) == loss_name(late)
    assert output_activation(early) == output_activation(late) == \
        tensorflow.keras.activations.sigmoid

    # train both networks
    late_fit_kwargs.update(verbose=0, epochs=50)
    late_hist = late.fit(train_in, train_out, **late_fit_kwargs)
    early_fit_kwargs.update(verbose=0, epochs=50,
                            validation_data=(test_in, test_out))
    early_hist = early.fit(train_in, train_out, **early_fit_kwargs)

    # check that accuracy levels are acceptable
    all1_accuracy = np.sum(test_out == 1) / test_out.size
    early_accuracy = binary_accuracy(early.predict(test_in), test_out)
    late_accuracy = binary_accuracy(late.predict(test_in), test_out)
    assert early_accuracy > 0.75
    assert late_accuracy > 0.75
    with capsys.disabled():
        accuracy_format = "{1:.1%} accuracy for {0} on census income".format
        print()
        print(accuracy_format("baseline", all1_accuracy))
        print(accuracy_format("early", early_accuracy))
        print(accuracy_format("late", late_accuracy))
    assert early_accuracy > all1_accuracy
    assert late_accuracy > all1_accuracy

    # check that the first network stopped early (fewer epochs)
    assert len(early_hist.history["loss"]) < len(late_hist.history["loss"])


def load_hdf5(path):
    with h5py.File(path, 'r') as f:
        train = f["train"]
        train_out = np.array(train["output"])
        train_in = np.array(train["input"])
        test = f["test"]
        test_out = np.array(test["output"])
        test_in = np.array(test["input"])
    return train_in, train_out, test_in, test_out


def assert_layers_equal(layers1: List[tensorflow.keras.layers.Layer],
                        layers2: List[tensorflow.keras.layers.Layer]):
    def layer_info(layer):
        return (layer.__class__,
                getattr(layer, "units", None),
                getattr(layer, "activation", None))

    assert [layer_info(l) for l in layers1] == [layer_info(l) for l in layers2]


def assert_compile_parameters_equal(model1: tensorflow.keras.models.Model,
                                    model2: tensorflow.keras.models.Model):
    def to_dict(obj):
        items = dict(__class__=obj.__class__.__name__, **vars(obj))
        to_remove = {key for key, value in items.items() if key.endswith("_fn")}
        for key in to_remove:
            items.pop(key)

    assert to_dict(model1.optimizer) == to_dict(model2.optimizer)


def loss_name(model):
    if isinstance(model.loss, str):
        loss = getattr(tensorflow.keras.losses, model.loss)
    else:
        loss = model.loss
    return loss.__name__.lower()


def hidden_activations(model):
    return [layer.activation
            for layer in model.layers[:-1] if hasattr(layer, "activation")]


def output_activation(model):
    return model.layers[-1].activation


def root_mean_squared_error(system: np.ndarray, human: np.ndarray):
    return ((system - human) ** 2).mean(axis=0) ** 0.5


def multi_class_accuracy(system: np.ndarray, human: np.ndarray):
    return np.mean(np.argmax(system, axis=1) == np.argmax(human, axis=1))


def binary_accuracy(system: np.ndarray, human: np.ndarray):
    return np.mean(np.round(system) == human)
