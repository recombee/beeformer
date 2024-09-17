import os

os.environ["KERAS_BACKEND"] = "torch"

import argparse
import subprocess

from time import time

parser = argparse.ArgumentParser()

parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--device", default=None, type=str, help="Limit device to run on")
parser.add_argument("--flag", default="none", type=str, help="flag for distinction of experiments, default none")

# dataset
parser.add_argument("--dataset", default="-", type=str, help="Dataset to run on")

# sentence transformer details
parser.add_argument("--sbert", default="none", type=str, help="Input sentence transformer model to train")
parser.add_argument("--max_seq_length", default=0, type=int, help="Maximum sequece length for sbert")


args = parser.parse_args([] if "__file__" not in globals() else None)
print(args)

if args.device is not None:
    print(f"Limiting devices to {args.device}")
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.device}"

import keras
import math
import numpy as np
import torch

from models import SparseKerasELSA
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config import config
from utils import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {DEVICE}")


def main(args):
    # prepare logging folder
    folder = f"results/{str(pd.Timestamp('today'))} {9*int(1e6)+np.random.randint(999999)}".replace(" ", "_")
    if not os.path.exists(folder):
        os.makedirs(folder)
    vargs = vars(args)
    vargs["cuda_or_cpu"] = DEVICE
    pd.Series(vargs).to_csv(f"{folder}/setup.csv")
    print(folder)

    # set random seeds for reproducibility
    torch.manual_seed(args.seed)
    keras.utils.set_random_seed(args.seed)
    np.random.seed(args.seed)

    # read data
    if args.dataset not in config.keys():
        print("Unknown dataset. List of available datsets: \n")
        for x in config.keys():
            print(x)
        return

    dataset, params = config[args.dataset]
    dataset.load_interactions(**params)
    csev = TimeBasedEvaluation(dataset)

    print(dataset)

    sbert = SentenceTransformer(args.sbert, device=DEVICE, trust_remote_code=True)
    if args.max_seq_length > 0:
        sbert.max_seq_length = args.max_seq_length
    embs = sbert.encode(dataset.texts, show_progress_bar=True)

    model = SparseKerasELSA(
        len(dataset.all_interactions.item_id.cat.categories),
        embs.shape[1],
        dataset.all_interactions.item_id.cat.categories,
        device=DEVICE,
    )
    model.to(DEVICE)
    model.set_weights([embs])

    df_preds = model.predict_df(csev.test_src)  # , candidates_df=csev.candidates_df)
    results = csev(df_preds)
    print(results)

    # final logs
    pd.Series(results).to_csv(f"{folder}/result.csv")
    print("results file written")

    pd.Series(0).to_csv(f"{folder}/timer.csv")
    print("timer written")


if __name__ == "__main__":
    main(args)
