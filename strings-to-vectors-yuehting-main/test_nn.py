import sys

import numpy as np
import pytest

import nn


def assert_array_equal(actual, desired):
    assert type(actual) is np.ndarray
    np.testing.assert_array_equal(actual, desired)


@pytest.mark.timeout(1)
def test_indexes():
    vocab = ["four", "three", "", "two", "one"]
    objects = ["one", "", "four", "four"]
    indexes = np.array([4, 2, 0, 0])

    index = nn.Index(vocab)
    assert_array_equal(index.objects_to_indexes(objects), indexes)
    assert index.indexes_to_objects(indexes) == objects

    index = nn.Index(vocab, start=1)
    assert_array_equal(index.objects_to_indexes(objects), indexes + 1)
    assert index.indexes_to_objects(indexes + 1) == objects


@pytest.mark.timeout(1)
def test_duplicates_in_vocabulary():
    vocab = [1, 4, 3, 1, 1, 2, 5, 2, 4]
    objects = [1, 2, 3, 4]
    indexes = np.array([0, 3, 2, 1])

    index = nn.Index(vocab)
    assert_array_equal(index.objects_to_indexes(objects), indexes)
    assert index.indexes_to_objects(indexes) == objects

    index = nn.Index(vocab, start=2)
    assert_array_equal(index.objects_to_indexes(objects), indexes + 2)
    assert index.indexes_to_objects(indexes + 2) == objects


@pytest.mark.timeout(1)
def test_out_of_vocabulary():
    vocab = "abcd"
    objects = "bcde"
    indexes = np.array([1, 2, 3, -1])
    indexes2 = np.array([1, 2, 3, 4])

    index = nn.Index(vocab)
    assert_array_equal(index.objects_to_indexes(objects), indexes)
    assert index.indexes_to_objects(indexes) == ["b", "c", "d"]
    assert index.indexes_to_objects(indexes2) == ["b", "c", "d"]

    index = nn.Index(vocab, start=3)
    assert_array_equal(index.objects_to_indexes(objects), indexes + 3)
    assert index.indexes_to_objects(indexes + 3) == ["b", "c", "d"]
    assert index.indexes_to_objects(indexes2 + 3) == ["b", "c", "d"]


@pytest.mark.timeout(1)
def test_binary_vector():
    vocab = "she sells seashells by the seashore".split()
    objects = "the seashells she sells".split()
    sorted_objects = "she sells seashells the".split()
    vector = np.array([1, 1, 1, 0, 1, 0])

    index = nn.Index(vocab)
    assert_array_equal(index.objects_to_binary_vector(objects), vector)
    assert index.binary_vector_to_objects(vector) == sorted_objects
    assert_array_equal(index.objects_to_binary_vector([]), np.zeros(6))
    assert index.binary_vector_to_objects(np.zeros(6)) == []

    index = nn.Index(vocab, start=4)
    shifted_vector = np.array([0, 0, 0, 0, 1, 1, 1, 0, 1, 0])
    assert_array_equal(index.objects_to_binary_vector(objects), shifted_vector)
    assert index.binary_vector_to_objects(shifted_vector) == sorted_objects
    assert_array_equal(index.objects_to_binary_vector([]), np.zeros(10))
    assert index.binary_vector_to_objects(np.zeros(10)) == []


@pytest.mark.timeout(1)
def test_index_matrix():
    vocab = "abcdef"
    objects = [["a", "b", "c"],
               ["g", "e", "d"]]
    matrix = np.array([[0, 1, 2],
                       [-1, 4, 3]])

    index = nn.Index(vocab)
    assert_array_equal(index.objects_to_index_matrix(objects), matrix)
    assert index.index_matrix_to_objects(matrix) == [["a", "b", "c"],
                                                     ["e", "d"]]

    index = nn.Index(vocab, start=2)
    assert_array_equal(index.objects_to_index_matrix(objects), matrix + 2)
    assert index.index_matrix_to_objects(matrix + 2) == [["a", "b", "c"],
                                                         ["e", "d"]]


@pytest.mark.timeout(1)
def test_ragged_index_matrix():
    vocab = "abcdefghijk"
    objects = [["a", "b", "c"],
               ["e"],
               ["f", "g", "h", "i"],
               ["j", "k"]]
    matrix = np.array([[1, 2, 3, 0],
                       [5, 0, 0, 0],
                       [6, 7, 8, 9],
                       [10, 11, 0, 0]])

    index = nn.Index(vocab, start=1)
    assert_array_equal(index.objects_to_index_matrix(objects), matrix)
    assert index.index_matrix_to_objects(matrix) == objects

    index = nn.Index(vocab)
    assert_array_equal(index.objects_to_index_matrix(objects), matrix - 1)
    assert index.index_matrix_to_objects(matrix - 1) == objects


@pytest.mark.timeout(1)
def test_binary_matrix():
    vocab = "abcdef"
    objects = [["a", "b", "c"],
               ["e"]]
    matrix = np.array([[1, 1, 1, 0, 0, 0],
                       [0, 0, 0, 0, 1, 0]])

    index = nn.Index(vocab)
    assert_array_equal(index.objects_to_binary_matrix(objects), matrix)
    assert index.binary_matrix_to_objects(matrix) == objects

    index = nn.Index(vocab, start=3)
    shifted_matrix = np.array([[0, 0, 0, 1, 1, 1, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 1, 0]])
    assert_array_equal(index.objects_to_binary_matrix(objects), shifted_matrix)
    assert index.binary_matrix_to_objects(shifted_matrix) == objects


@pytest.mark.timeout(2)
def test_large_sequences():
    vocab = [chr(i) for i in range(sys.maxunicode + 1)]
    objects = list('schön día \U0010ffff' * 100)

    index = nn.Index(vocab)
    vector = index.objects_to_binary_vector(objects)
    assert vector[ord('í')] == 1
    assert vector[ord('\U0010ffff')] == 1
    assert vector[ord('o')] == 0
    assert index.binary_vector_to_objects(vector) == list(' acdhnsíö\U0010ffff')
