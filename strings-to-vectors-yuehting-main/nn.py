"""
The main code for the Strings-to-Vectors assignment. See README.md for details.
"""
from typing import Sequence, Any

import numpy as np


class Index:
    """
    Represents a mapping from a vocabulary (e.g., strings) to integers.
    """

    def __init__(self, vocab: Sequence[Any], start=0):
        """
        Assigns an index to each unique item in the `vocab` iterable,
        with indexes starting from `start`.

        Indexes should be assigned in order, so that the first unique item in
        `vocab` has the index `start`, the second unique item has the index
        `start + 1`, etc.
        """
        self.object_index = {}
        self.start = start

        for i in vocab:
            if i not in self.object_index:
                self.object_index[i] = start
                start = start + 1

    def objects_to_indexes(self, object_seq: Sequence[Any]) -> np.ndarray:
        """
        Returns a vector of the indexes associated with the input objects.

        For objects not in the vocabulary, `start-1` is used as the index.

        :param object_seq: A sequence of objects.
        :return: A 1-dimensional array of the object indexes.
        """

        return np.array([self.object_index[object] if object in self.object_index else self.start-1
                         for object in object_seq])

    def objects_to_index_matrix(
            self, object_seq_seq: Sequence[Sequence[Any]]) -> np.ndarray:
        """
        Returns a matrix of the indexes associated with the input objects.

        For objects not in the vocabulary, `start-1` is used as the index.

        If the sequences are not all of the same length, shorter sequences will
        have padding added at the end, with `start-1` used as the pad value.

        :param object_seq_seq: A sequence of sequences of objects.
        :return: A 2-dimensional array of the object indexes.
        """

        max_len = max(len(item) for item in object_seq_seq)
        new_objects = np.full((len(object_seq_seq), max_len), self.start-1)

        for i, object_seq in enumerate(object_seq_seq):
            result = self.objects_to_indexes(object_seq)
            new_objects[i, :len(result)] = result

        return new_objects

    def objects_to_binary_vector(self, object_seq: Sequence[Any]) -> np.ndarray:
        """
        Returns a binary vector, with a 1 at each index corresponding to one of
        the input objects.

        :param object_seq: A sequence of objects.
        :return: A 1-dimensional array, with 1s at the indexes of each object,
                 and 0s at all other indexes.
        """

        result = np.zeros(len(self.object_index) + self.start)
        for item in object_seq:
            if item in self.object_index:
                result[self.object_index[item]] = 1

        return result

    def objects_to_binary_matrix(
            self, object_seq_seq: Sequence[Sequence[Any]]) -> np.ndarray:
        """
        Returns a binary matrix, with a 1 at each index corresponding to one of
        the input objects.

        :param object_seq_seq: A sequence of sequences of objects.
        :return: A 2-dimensional array, where each row in the array corresponds
                 to a row in the input, with 1s at the indexes of each object,
                 and 0s at all other indexes.
        """

        return np.stack([self.objects_to_binary_vector(object_seq)
                         for object_seq in object_seq_seq])

    def indexes_to_objects(self, index_vector: np.ndarray) -> Sequence[Any]:
        """
        Returns a sequence of objects associated with the indexes in the input
        vector.

        If, for any of the indexes, there is not an associated object, that
        index is skipped in the output.

        :param index_vector: A 1-dimensional array of indexes
        :return: A sequence of objects, one for each index.
        """

        return [list(self.object_index.keys())[list(self.object_index.values()).index(i)]
                for i in index_vector if i in self.object_index.values()]

    def index_matrix_to_objects(
            self, index_matrix: np.ndarray) -> Sequence[Sequence[Any]]:
        """
        Returns a sequence of sequences of objects associated with the indexes
        in the input matrix.

        If, for any of the indexes, there is not an associated object, that
        index is skipped in the output.

        :param index_matrix: A 2-dimensional array of indexes
        :return: A sequence of sequences of objects, one for each index.
        """

        return [self.indexes_to_objects(i) for i in index_matrix]

    def binary_vector_to_objects(self, vector: np.ndarray) -> Sequence[Any]:
        """
        Returns a sequence of the objects identified by the nonzero indexes in
        the input vector.

        If, for any of the indexes, there is not an associated object, that
        index is skipped in the output.

        :param vector: A 1-dimensional binary array
        :return: A sequence of objects, one for each nonzero index.
        """

        (indexarray,) = np.nonzero(vector)
        return [list(self.object_index.keys())[list(self.object_index.values()).index(i)]
                for i in indexarray if i in self.object_index.values()]

    def binary_matrix_to_objects(
            self, binary_matrix: np.ndarray) -> Sequence[Sequence[Any]]:
        """
        Returns a sequence of sequences of objects identified by the nonzero
        indices in the input matrix.

        If, for any of the indexes, there is not an associated object, that
        index is skipped in the output.

        :param binary_matrix: A 2-dimensional binary array
        :return: A sequence of sequences of objects, one for each nonzero index.
        """

        return [self.binary_vector_to_objects(index) for index in binary_matrix]
