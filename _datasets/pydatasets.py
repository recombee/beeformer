import os
import torch

os.environ["KERAS_BACKEND"] = "torch"

import keras

from keras import backend
from keras import ops
from keras.src.backend.torch.core import *

import scipy.sparse

import math

from datasets.utils import *


class BasicRecSysDataset(keras.utils.PyDataset):
    """
    input sparse interaction matrix
    output batches of user vectors input, output (output==input)
    """

    def __init__(
        self,
        X: scipy.sparse.csr_matrix,
        batch_size: int = 128,
        shuffle=False,
        workers=1,
        use_multiprocessing=False,
        max_queue_size=10,
    ):
        super().__init__(workers=workers, use_multiprocessing=use_multiprocessing, max_queue_size=max_queue_size)
        self.X, self.batch_size, self.shuffle = X, batch_size, shuffle
        self.indices = np.arange(X.shape[0])
        if self.shuffle:
            self.on_epoch_end()

    def __len__(self):
        # Return number of batches.
        return math.ceil(self.X.shape[0] / (self.batch_size))

    def __getitem__(self, n):
        ind = n * self.batch_size
        ind_min = ind
        ind_max = ind + self.batch_size
        slicer = self.indices[ind_min:ind_max]
        M = self.X[slicer]
        # R = torch.from_numpy(M.toarray().astype("float32")).cuda()
        R = M.toarray().astype("float32")
        return R, R

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


class PredictDfRecSysDataset(keras.utils.PyDataset):
    """
    input sparse interaction matrix + item_ids to know order of items
    output batches of user vectors and user ids
    """

    def __init__(self, df, item_ids, batch_size=128, workers=1, use_multiprocessing=False, max_queue_size=10):
        super().__init__(workers=workers, use_multiprocessing=use_multiprocessing, max_queue_size=max_queue_size)
        self.user_ids = np.array(df.user_id.cat.categories)
        self.df, self.batch_size, self.items_ids = df, batch_size, item_ids
        self.X = get_sparse_matrix_from_dataframe(df, item_indices=self.items_ids)

    def __len__(self):
        # Return number of batches.
        return math.ceil(self.X.shape[0] / (self.batch_size))

    def __getitem__(self, n):
        ind = n * self.batch_size
        ind_min = ind
        ind_max = ind + self.batch_size
        M = self.X[ind_min:ind_max]
        # R = torch.from_numpy(M.toarray().astype("float32")).cuda()
        R = M.toarray().astype("float32")
        return R, self.user_ids[ind_min:ind_max]


class SparseRecSysDataset(keras.utils.PyDataset):
    """
    input sparse interaction matrix
    output batches of user vectors + slicer with indices of nonzero columns
    """

    def __init__(
        self,
        X: scipy.sparse.csr_matrix,
        batch_size: int = 32,
        shuffle=False,
        workers=1,
        use_multiprocessing=False,
        max_queue_size=10,
    ):
        super().__init__(workers=workers, use_multiprocessing=use_multiprocessing, max_queue_size=max_queue_size)
        self.X, self.batch_size, self.shuffle = X, batch_size, shuffle
        self.indices = np.arange(X.shape[0])
        if self.shuffle:
            self.on_epoch_end()

    def __len__(self):
        # Return number of batches.
        return math.ceil(self.X.shape[0] / (self.batch_size))

    def __getitem__(self, n):
        ind = n * self.batch_size
        ind_min = ind
        ind_max = ind + self.batch_size
        slicer = self.indices[ind_min:ind_max]
        M = self.X[slicer]
        # R = torch.from_numpy(M.toarray().astype("float32")).cuda()
        item_slicer = np.where(M.getnnz(0) > 0)[0]
        R = M.toarray().astype("float32")
        return torch.from_numpy(R), torch.from_numpy(item_slicer).long()

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


class SparseTransposedRecSysDataset(keras.utils.PyDataset):
    """
    input X=sparse interaction matrix, arange=reordering of items in X, min_support=number of interactions in row to keep
    output batches of user vectors + slicer with indices of nonzero columns
    """

    def __init__(
        self,
        X: scipy.sparse.csr_matrix,
        batch_size: int = 32,
        arange=None,
        shuffle=False,
        workers=1,
        use_multiprocessing=False,
        max_queue_size=10,
        min_support=2.0,
    ):
        super().__init__(workers=workers, use_multiprocessing=use_multiprocessing, max_queue_size=max_queue_size)
        self.X, self.batch_size, self.shuffle = X, batch_size, shuffle
        self.min_support = min_support
        if arange is None:
            self.indices = np.arange(X.shape[1])
        else:
            self.indices = arange
        if self.shuffle:
            self.on_epoch_end()

    def __len__(self):
        # Return number of batches.
        return math.ceil(self.X.shape[1] / (self.batch_size // 2))

    def __getitem__(self, n):
        ind = n * self.batch_size // 2
        ind_min = max(ind - self.batch_size // 2, 0)
        ind_max = ind + self.batch_size // 2
        slicer = self.indices[ind_min:ind_max]
        M = self.X[:, slicer]
        # R = torch.from_numpy(M.toarray().astype("float32")).cuda()
        M = M[M.getnnz(1) > 0]
        M = M.toarray().astype("float32")
        M = M[M.sum(1) >= self.min_support]
        R = torch.from_numpy(M)
        return R, torch.from_numpy(slicer).long()

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


class __SparseRecSysDatasetWithNegatives(keras.utils.PyDataset):
    """
    input sparse interaction matrix
    output batches of user vectors + slicer with indices of nonzero columns
    """

    def __init__(
        self,
        X: scipy.sparse.csr_matrix,
        device,
        batch_size: int = 32,
        shuffle=False,
        workers=1,
        use_multiprocessing=False,
        max_queue_size=10,
        max_output=None,
    ):
        super().__init__(workers=workers, use_multiprocessing=use_multiprocessing, max_queue_size=max_queue_size)
        self.X, self.batch_size, self.shuffle = X, batch_size, shuffle
        self.indices = np.arange(X.shape[0])
        self.items_indices = np.arange(X.shape[1])
        self.device = device
        if max_output is None:
            self.max_output = X.shape[1]
        else:
            self.max_output = max_output

        if self.shuffle:
            self.on_epoch_end()

    def __len__(self):
        # Return number of batches.
        return math.ceil(self.X.shape[0] / (self.batch_size))

    def __getitem__(self, n):
        ind = n * self.batch_size
        ind_min = ind
        ind_max = ind + self.batch_size
        slicer = self.indices[ind_min:ind_max]
        M = self.X[slicer]
        # R = torch.from_numpy(M.toarray().astype("float32")).cuda()
        item_slicer = np.where(M.getnnz(0) > 0)[0]
        mask = np.ones(self.items_indices.shape, dtype=bool)
        mask[item_slicer] = False
        # todo - different approach, always have at least 1 neg sample
        if self.max_output - len(item_slicer) > 0:
            item_slicer_with_negatives = np.hstack(
                [item_slicer, np.random.choice(self.items_indices[mask], self.max_output - len(item_slicer))]
            )
        else:
            item_slicer_with_negatives = item_slicer

        # R = M.toarray().astype("float32")
        scipy_coo = M.tocoo()

        scipy_coo_x = M[:, item_slicer].tocoo()
        # todo: different approach, send x and only negatives, then construct y = stack(y, negatives) in trainstep
        scipy_coo_y = M[:, item_slicer_with_negatives].tocoo()
        torch_coo_x = torch.sparse_coo_tensor(
            np.vstack([scipy_coo_x.row, scipy_coo_x.col]),
            scipy_coo_x.data.astype(np.float32),
            scipy_coo_x.shape,
        )
        torch_coo_y = torch.sparse_coo_tensor(
            np.vstack([scipy_coo_y.row, scipy_coo_y.col]),
            scipy_coo_y.data.astype(np.float32),
            scipy_coo_y.shape,
        )
        return (torch_coo_x.to(self.device).to_dense(), torch_coo_y.to(self.device).to_dense()), (
            torch.from_numpy(item_slicer).long(),
            torch.from_numpy(item_slicer_with_negatives).long(),
        )

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


class SparseRecSysDatasetWithNegatives(keras.utils.PyDataset):
    """
    input sparse interaction matrix
    output batches of user vectors + slicer with indices of nonzero columns
    """

    def __init__(
        self,
        X: scipy.sparse.csr_matrix,
        device,
        batch_size: int = 32,
        shuffle=False,
        workers=1,
        use_multiprocessing=False,
        max_queue_size=10,
        max_output=None,
    ):
        super().__init__(workers=workers, use_multiprocessing=use_multiprocessing, max_queue_size=max_queue_size)
        self.X, self.batch_size, self.shuffle = X, batch_size, shuffle
        self.indices = np.arange(X.shape[0])
        self.items_indices = np.arange(X.shape[1])
        self.device = device
        if max_output is None:
            self.max_output = X.shape[1]
        else:
            self.max_output = max_output

        if self.shuffle:
            self.on_epoch_end()

    def __len__(self):
        # Return number of batches.
        return math.ceil(self.X.shape[0] / (self.batch_size))

    def __getitem__(self, n):
        ind = n * self.batch_size
        ind_min = ind
        ind_max = ind + self.batch_size
        slicer = self.indices[ind_min:ind_max]
        M = self.X[slicer]
        # R = torch.from_numpy(M.toarray().astype("float32")).cuda()
        item_slicer = np.where(M.getnnz(0) > 0)[0]
        mask = np.ones(self.items_indices.shape, dtype=bool)
        mask[item_slicer] = False
        # todo - different approach, always have at least 1 neg sample
        num_negatives = max(1, self.max_output - len(item_slicer))

        item_slicer_for_negatives = np.random.choice(self.items_indices[mask], num_negatives)
        item_slicer_with_negatives = np.hstack([item_slicer, item_slicer_for_negatives])
        # R = M.toarray().astype("float32")
        scipy_coo = M.tocoo()

        scipy_coo_x = M[:, item_slicer].tocoo()

        # todo: different approach, send x and only negatives,
        # then construct y = stack(y, negatives) in trainstep

        scipy_coo_y = M[:, item_slicer_for_negatives].tocoo()
        torch_coo_x = torch.sparse_coo_tensor(
            np.vstack([scipy_coo_x.row, scipy_coo_x.col]),
            scipy_coo_x.data.astype(np.float32),
            scipy_coo_x.shape,
        )
        torch_coo_y = torch.sparse_coo_tensor(
            np.vstack([scipy_coo_y.row, scipy_coo_y.col]),
            scipy_coo_y.data.astype(np.float32),
            scipy_coo_y.shape,
        )
        return (torch_coo_x.to(self.device).to_dense(), torch_coo_y.to(self.device).to_dense()), (
            torch.from_numpy(item_slicer).long(),
            torch.from_numpy(item_slicer_with_negatives).long(),
        )

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
