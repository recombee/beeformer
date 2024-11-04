import os

os.environ["KERAS_BACKEND"] = "torch"

import keras
import math
import numpy as np
import scipy.sparse
import torch

from utils import *

class beeformerDataset(keras.utils.PyDataset):
    """
    input sparse interaction matrix
    output batches of user vectors + slicer with indices of nonzero columns
    """

    def __init__(
        self,
        X: scipy.sparse.csr_matrix,
        tokenized_sentences,
        device,
        batch_size: int = 1024,
        shuffle=False,
        workers=1,
        use_multiprocessing=False,
        max_queue_size=10,
        max_output=None,
    ):
        super().__init__(workers=workers, use_multiprocessing=use_multiprocessing, max_queue_size=max_queue_size)
        self.X, self.batch_size, self.shuffle, self.tokenized_sentences = (
            X,
            batch_size,
            shuffle,
            {k: v for k, v in tokenized_sentences.items()},
        )

        assert get_first_item(tokenized_sentences).shape[0] == X.shape[1]
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

        tokenized_items = {k: v[item_slicer_with_negatives].to(self.device) for k, v in self.tokenized_sentences.items()}

        slicer = np.arange(len(item_slicer))
        slicer_neg = np.arange(len(item_slicer_with_negatives))

        return (torch_coo_x.to(self.device).to_dense(), torch_coo_y.to(self.device).to_dense()), (
            tokenized_items,
            torch.from_numpy(slicer).long(),
            torch.from_numpy(slicer_neg).long(),
        )

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
