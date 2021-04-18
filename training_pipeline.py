from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_absolute_error

from preprocessing_utils import word_count_score, char_count_score, over_18, readability_score\
                                , sentiment_score, toxicity_score, subjectivity_score, day_from_date\
                                , upper_case_score, process_df


import joblib

def feature_engineering(df):
    df = word_count_score(df)
    df = char_count_score(df)
    df = over_18(df)
    df = readability_score(df)
    df = sentiment_score(df)
    df = toxicity_score(df)
    df = subjectivity_score(df)
    df = day_from_date(df)
    df = upper_case_score(df)
    x, y = process_df(df)
    return x, y


def up_votes_training_pipeline(x,y, regressor=MLPRegressor(solver='adam', alpha=1e-5, hidden_layer_sizes=(128, 64, 32))):
    numeric_features = x.select_dtypes(include=['int16', 'int32','int64', 'float64']).columns
    y_numeric = y.select_dtypes(include=['int16', 'int32','int64', 'float64']).columns
    numeric_features.extend(y_numeric)

    numeric_transformer = Pipeline(steps=[('scaler', RobustScaler())])

    preprocessor = ColumnTransformer(transformers=
                                    [
                                    ('numeric', numeric_transformer, numeric_features),
                                    ])

    pipeline = Pipeline(steps = [
               ('preprocessor', preprocessor)
              ,('regressor',regressor)
           ])

    return pipeline

def train_upvotes(df, regressor=MLPRegressor(solver='adam', alpha=1e-5, hidden_layer_sizes=(128, 64, 32)), regressor_path='regression_model.pkl', fine_tune=False):
    print("Feature Engineering The DataFrame")
    x, y = feature_engineering(df)
    print("\n---------------------------\n")

    if fine_tune:
        print(f"Loading The Model From {regressor_path}")
        rf_model = joblib.load(regressor_path)
        print("Model Loaded Successfully")
        print("\n---------------------------\n")
        print("Training The Regression Model....")
        rf_model.fit(x, y)
        print("Model Trained Successfully")

    else:
        print("Building Regression Model")
        rf_model = up_votes_training_pipeline(x, y, regressor)
        print("\n---------------------------\n")
        print("Training Regression Model")
        rf_model.fit(x, y)
        print("Model Trained Successfully")

    joblib.dump(rf_model, 'regression_model.pkl')
    return rf_model

def evaluate_model(x_test, y_test, path='regression_model.pkl'):
    print(f"Loading The Model From {path}")
    rf_model = joblib.load(path)
    print("Model Loaded Successfully")
    print("\n---------------------------\n")
    print("Evaluating Regression Model")
    preds = rf_model.predict(x_test)
    print(f"Your model r2_score is {r2_score(preds, y_test)}")
    print(f"Your model mean abolute error is {mean_absolute_error(preds, y_test)}")










    





