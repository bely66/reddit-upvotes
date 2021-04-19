import pandas as pd
import numpy as np
from training_pipeline import train_upvotes, evaluate_model

datafile = "Eluvio_DS_Challenge.csv"
chunksize = 100
fine_tune = False
for i, df in enumerate(pd.read_csv(datafile, chunksize=chunksize, usecols=["up_votes", "title", "author", "date_created","over_18"])):
  msk = np.random.rand(len(df)) < 0.8
  train = df[msk]
  test = df[~msk]
  train_upvotes(train, i, fine_tune=fine_tune)
  evaluate_model(test)
  fine_tune = True