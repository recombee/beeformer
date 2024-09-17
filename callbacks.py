import os

os.environ["KERAS_BACKEND"] = "torch"

import keras
import pandas as pd

from models import SparseKerasELSA
from time import time
from utils import *


class evaluateWriter(keras.callbacks.Callback):
    def __init__(
        self,
        items_idx,
        sbert,
        texts,
        evaluator,
        logdir,
        DEVICE,
        sbert_name="sbert_temp_model",
        evaluate_epoch="false",
        save_every_epoch="false",
    ):
        super().__init__()
        self.evaluator = evaluator
        self.logdir = logdir
        self.sbert = sbert
        self.texts = texts
        self.items_idx = items_idx
        self.DEVICE = DEVICE
        self.results_list = []
        self.sbert_name = sbert_name
        self.evaluate_epoch = evaluate_epoch
        self.save_every_epoch = save_every_epoch

    def on_epoch_end(self, epoch, logs=None):
        print()
        if self.save_every_epoch == "true":
            print("saving sbert model")
            self.sbert.save(f"{self.sbert_name}-epoch-{epoch}")
        if self.evaluate_epoch == "true":
            embs = self.sbert.encode(self.texts, show_progress_bar=True)
            model = SparseKerasELSA(len(self.items_idx), embs.shape[1], self.items_idx, device=self.DEVICE)
            model.to(self.DEVICE)
            model.set_weights([embs])
            if isinstance(self.evaluator, ColdStartEvaluation):
                df_preds = model.predict_df(
                    self.evaluator.test_src,
                    candidates_df=(
                        self.evaluator.cold_start_candidates_df
                        if hasattr(self.evaluator, "cold_start_candidates_df")
                        else None
                    ),
                    k=1000,
                )
                df_preds = df_preds[
                    ~df_preds.set_index(["item_id", "user_id"]).index.isin(
                        self.evaluator.test_src.set_index(["item_id", "user_id"]).index
                    )
                ]
            else:
                df_preds = model.predict_df(self.evaluator.test_src)
            results = self.evaluator(df_preds)

            # this should be logged to tensorboard after every epoch but tensorboard does not work correctly in keras 3 with torch backend
            print(results)
            pd.Series(results).to_csv(f"{self.logdir}/result-epoch-{epoch}.csv")
            print("results file written")
            self.results_list.append(results)
