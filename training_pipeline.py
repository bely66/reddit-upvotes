from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_absolute_error

from preprocessing_utils import word_count_score, char_count_score, over_18, readability_score\
                                , sentiment_score, toxicity_score, subjectivity_score, day_from_date\
                                , upper_case_score, process_df

from MLP import MLPRegressor, TrainRegressor

import joblib

def feature_engineering(df):
    print("Adding Word Count Feature")
    df = word_count_score(df)
    print("Adding Char Count Feature")
    df = char_count_score(df)
    print("Adding Over 18 Feature")
    df = over_18(df)
    print("Adding Readability Score Feature")
    df = readability_score(df)
    print("Adding Sentiment Score Feature")
    df = sentiment_score(df)
    print("Adding Toxicity Score Feature")
    df = toxicity_score(df)
    print("Adding Subjectivity Score Feature")
    df = subjectivity_score(df)
    print("Adding Posted Day Feature")
    df = day_from_date(df)
    print("Adding Upper Score Ratio Feature")
    df = upper_case_score(df)
    print("Dropping Un-necessary Tables")
    x, y = process_df(df)
    return x, y


def up_votes_training_pipeline(input_shape, fine_tune=False):
    regressor = MLPRegressor(input_shape)

    return regressor
    # Define the loss function and optimizer
    



def train_upvotes(df, i, fine_tune=False):
    print("Feature Engineering The DataFrame")
    x, y = feature_engineering(df)
    print("\n---------------------------\n")
    
    if fine_tune:
        print(f"Loading The Model....")
        train_manager = TrainRegressor()
        print("Model Loaded Successfully")
        print("\n---------------------------\n")
        print("Training The Regression Model....")
        train_manager.fit(i, x, y)
        print("Model Trained Successfully")

    else:
        model = up_votes_training_pipeline(input_shape=x.shape[1])
        train_manager = TrainRegressor(mlp=model)
        print("Building Regression Model")
        print("\n---------------------------\n")
        print("Training Regression Model")
        train_manager.fit(i, x, y)
        print("Model Trained Successfully")

    return train_manager.model

def evaluate_model(df, path='regression_model.pkl'):
    x, y = feature_engineering(df)
    print(f"Loading The Model From {path}")
    rf_model = TrainRegressor(train=False)
    print("Model Loaded Successfully")
    print("\n---------------------------\n")
    print("Evaluating Regression Model")
    preds = rf_model.predict(x)
    print(f"Your model r2_score is {r2_score(preds, y)}")
    print(f"Your model mean abolute error is {mean_absolute_error(preds, y)}")










    





