import pandas as pd

ratings = pd.read_csv("Books.csv", header=None)
ratings.columns = ["asin", "uid", "rating", "timestamp"]
ratings = ratings[ratings.rating >= 4]
ratings["asin"] = ratings["asin"].astype("category")
ratings["uid"] = ratings["uid"].astype("category")

df = pd.read_feather("item_text_descriptions.feather")
df = df.drop_duplicates("asin")
df = df[df.asin.isin(ratings.asin)]
df.to_feather("item_text_descriptions.feather")

ratings = ratings[ratings.asin.isin(df.asin)]
ratings.to_feather("ratings.feather")
