import pandas as pd
import numpy as np
from training_pipeline import train_upvotes, evaluate_model

datafile = "Eluvio_DS_Challenge.csv"
chunksize = 1000
fine_tune = False
manager = None
for epoch in range(5):
    print("Training epoch: ", epoch+1)
    print("\n---------------------------\n")

    for i, df in enumerate(pd.read_csv(datafile, chunksize=chunksize, usecols=["up_votes", "title", "author", "date_created","over_18"])):
        msk = np.random.rand(len(df)) < 0.8
        train = df[msk]
        test = df[~msk]

        print("Training batch: ", i+1)
        print("\n---------------------------\n")
        manager = train_upvotes(train, i, fine_tune=fine_tune, model=manager, epochs=1)
        print("Evaluating batch: ", i+1)
        print("\n---------------------------\n")
        evaluate_model(test, manager= manager)
        fine_tune = True