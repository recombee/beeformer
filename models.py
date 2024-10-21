import os

os.environ["KERAS_BACKEND"] = "torch"

import keras
import math
import numpy as np
import torch

from dataloaders import *
from layers import *


# beeformer optimized with nmse (expected loss since all the normalizations inside the train step)
class NMSEbeeformer(keras.models.Model):
    def __init__(self, tokenized_sentences, items_idx, sbert, device, top_k=0, sbert_batch_size=128):
        super().__init__()
        self.device = device
        self.sbert = LayerSBERT(sbert, device)
        self.items_idx = items_idx
        self.tokenized_sentences = tokenized_sentences
        self.top_k = top_k
        self.sbert_batch_size = sbert_batch_size

    def call(self, x):
        return self.sbert(x)

    def train_step(self, data):
        # Unpack the data
        a, b = data
        x, y = a
        y = torch.hstack((x, y))
        x_out = y
        tokenized_items, slicer, negative_slicer = b
        slicer = slicer.to(self.device)
        if negative_slicer is not None:
            negative_slicer = negative_slicer.to(self.device)

        # init everything for training
        self.zero_grad()
        sbert_batch_size = self.sbert_batch_size
        len_sentences = tokenized_items["input_ids"].shape[0]
        max_i = math.ceil(len_sentences / sbert_batch_size)

        # sbert forward pass #1 - we want to get embeddings for items to compute loss
        with torch.no_grad():
            # we are doing it in batches because of memory
            batched_results = []
            for i in range(max_i):
                ind = i * sbert_batch_size
                ind_min = ind
                ind_max = ind + sbert_batch_size
                batch_result = self.sbert({k: v[ind_min:ind_max] for k, v in tokenized_items.items()})
                batched_results.append(batch_result)
            A = torch.vstack(batched_results)

        # track gradients for A, this will be our gradient checkpoint
        A.requires_grad = True

        # compute ELSA forward pass only for rows with values
        A_slicer = A[slicer]
        A_slicer = torch.nn.functional.normalize(A_slicer, dim=-1)
        A_negative_slicer = A[negative_slicer]
        A_negative_slicer = torch.nn.functional.normalize(A_negative_slicer, dim=-1)
        A_slicer = A[slicer]
        A_slicer = torch.nn.functional.normalize(A_slicer, dim=-1)

        # ELSA step
        xA = torch.matmul(x, A_slicer)
        xAAT = torch.matmul(xA, A_negative_slicer.T)
        y_pred = keras.activations.relu(xAAT - x_out)

        # theoretically, this might improve performance for bigger dataset
        if self.top_k > 0:
            val, inds = torch.topk(y_pred, self.top_k)
            y = torch.gather(y, 1, inds)
            y_pred = val

        # compute loss
        loss = self.compute_loss(y=y, y_pred=y_pred)

        # compute gradients for the gradient checkpoint (our ELSA A matrix)
        loss.backward()

        # sbert forward pass #2
        # now we will do the sbert forward pass again, but this time we will track gradients this time, for memory reasons in again batches
        batched_results = []
        for i in range(max_i):
            ind = i * sbert_batch_size
            ind_min = ind
            ind_max = ind + sbert_batch_size
            # actual forward pass
            temp_out = self.sbert({k: v[ind_min:ind_max] for k, v in tokenized_items.items()})
            # we need to get gradients for part of A
            temp_out.retain_grad()
            # get the slice of corresponding gradients
            partial_A_grad = A.grad[ind_min:ind_max]
            # compute gradients for sbert
            temp_out.backward(gradient=partial_A_grad)

        # get gradients for sbert
        trainable_weights = [v for v in self.sbert.trainable_weights]
        gradients = [v.value.grad for v in trainable_weights]

        # Update weights
        with torch.no_grad():
            self.optimizer.apply(gradients, trainable_weights)

        # Update metrics (includes the metric that tracks the loss)
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)

        # Return a dict mapping metric names to current value
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}


# ELSA model optimized for sparse data, used only for predictions
class SparseKerasELSA(keras.models.Model):
    def __init__(self, n_items, n_dims, items_idx, device, top_k=0):
        super().__init__()
        self.device = device
        self.ELSA = LayerELSA(n_items, n_dims, device=device)
        self.items_idx = items_idx
        self.ELSA.build()
        self(np.zeros([1, n_items]))
        self.finetuning = False
        self.top_k = top_k

    def call(self, x):
        return self.ELSA(x)

    def train_step(self, data):
        # Unpack the data
        if len(data) == 2:
            full_x = None
            a, b = data
            x, y = a
            y = torch.hstack((x, y))
            slicer, negative_slicer = b

        elif len(data) == 3:
            full_x, slicer, negative_slicer = data
        else:
            full_x, slicer = data
            negative_slicer = None

        if full_x is not None:
            if negative_slicer is not None:
                y = full_x[:, negative_slicer]
            else:
                y = full_x

            x = full_x[:, slicer]

            x = x.to(self.device)
            y = y.to(self.device)

        x = torch.nn.functional.normalize(x, p=1.0, dim=-1)
        y = torch.nn.functional.normalize(y, p=1.0, dim=-1)

        x_out = y

        self.zero_grad()

        A = self.ELSA.A
        A_slicer = A[slicer]
        A_slicer = torch.nn.functional.normalize(A_slicer, dim=-1)

        if negative_slicer is not None:
            A_negative_slicer = A[negative_slicer]
            A_negative_slicer = torch.nn.functional.normalize(A_negative_slicer, dim=-1)
        else:
            A_negative_slicer = torch.nn.functional.normalize(A, dim=-1)

        xA = torch.matmul(x, A_slicer)
        xAAT = torch.matmul(xA, A_negative_slicer.T)
        y_pred = xAAT - x_out

        if self.finetuning:
            val, inds = torch.topk(y_pred, self.top_k)
            y = torch.gather(y, 1, inds)
            y_pred = val

        # Compute loss
        loss = self.compute_loss(y=y, y_pred=y_pred)

        # Call torch.Tensor.backward() on the loss to compute gradients
        # for the weights.
        loss.backward()

        trainable_weights = [v for v in self.trainable_weights]
        gradients = [v.value.grad for v in trainable_weights]

        # Update weights
        with torch.no_grad():
            self.optimizer.apply(gradients, trainable_weights)

        # Update metrics (includes the metric that tracks the loss)
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)

        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def predict_df(self, df, k=100, user_ids=None, candidates_df=None, block_reminder=True):
        # create predictions from data in dataframe, returns predictions in dataframe
        if user_ids is None:
            user_ids = np.array(df.user_id.cat.categories)

        if candidates_df is not None:
            candidates_vec = get_sparse_matrix_from_dataframe(candidates_df, item_indices=self.items_idx).toarray()
            candidates_vec = torch.from_numpy(candidates_vec)  # .to(self.device)

        data = PredictDfRecSysDataset(df, self.items_idx, batch_size=1024)

        dfs = []

        for i in tqdm(range(len(data)), total=len(data)):
            x, batch_uids = data[i]

            batch = torch.from_numpy(self.predict_on_batch(x))
            if block_reminder:
                mask = 1 - x.astype(bool)  # block reminder
                batch = batch * mask

            if candidates_df is not None:
                batch *= candidates_vec

            values_, indices_ = torch.topk(batch.to("cpu"), k)
            df = pd.DataFrame(
                {
                    "user_id": np.stack([batch_uids] * k).flatten("F"),
                    "item_id": np.array(self.items_idx)[indices_].flatten(),
                    "value": values_.flatten(),
                }
            )
            df["user_id"] = df["user_id"].astype(str).astype("category")
            df["item_id"] = df["item_id"].astype(str).astype("category")
            dfs.append(df)

        df = pd.concat(dfs)
        df["user_id"] = df["user_id"].astype(str).astype("category")
        df["item_id"] = df["item_id"].astype(str).astype("category")
        return df