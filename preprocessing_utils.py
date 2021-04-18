import re
import textstat
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import datetime
import calendar
from detoxify import Detoxify
from sklearn.preprocessing import RobustScaler


# each model takes in either a string or a list of strings


textstat.set_lang("en")
toxicity_model = Detoxify('original')

def subjective(text):
  words = ["i", "my"]
  count = 0
  text = text.lower().split()
  for word in words:
    count += text.count(word)

  return count

def weekday_from_date(date):
  year, month, day = date.split("-")
  day_number = datetime.date(day=int(day), month=int(month), year=int(year)).weekday()
  return calendar.day_name[day_number]

def uppercase_ratio(text):
  res = list(filter(lambda c: c.isupper(), text))
  ratio = len(res)/len(text.replace(" ", ""))
  return ratio


def subjectivity_score(df):
    df["subjectivity"] = df.title.apply(lambda x: subjective(x))
    return df

def readability_score(df):
    df["readability_index"] = df.title.apply(lambda x: textstat.automated_readability_index(x))

    return df

def sentiment_score(df):
    analyser = SentimentIntensityAnalyzer()
    df["neg_sentiment"] = df.title.apply(lambda x: analyser.polarity_scores(x).get("neg"))
    df["pos_sentiment"] = df.title.apply(lambda x: analyser.polarity_scores(x).get("pos"))
    df["neu_sentiment"] = df.title.apply(lambda x: analyser.polarity_scores(x).get("neu"))
    return df

def day_from_date(df):
    df["day_created"] = df.date_created.apply(lambda x: weekday_from_date(x))

    day_dummies = pd.get_dummies(df.day_created, prefix='day')
    df = pd.concat([df, day_dummies], axis=1)

    return df

def upper_case_score(df):
    df["uppercase_ratio"] = df.title.apply(lambda x: uppercase_ratio(x))
    return df

def toxicity_score(df):
    df["toxic"] = df.title.apply(lambda x: toxicity_model.predict(x).get("toxicity"))

    return df

def clean_data(df):
    # Load the regular expression library
    import re
    # Remove punctuation
    df.title = df.title.apply(lambda x: re.sub('[,\.!?]', '', x))

    return df
def get_category(votes):
  if votes < 10:
    return "low"
  elif votes > 10 and votes <120:
    return "moderate"
  
  elif votes > 120 and votes <500:
    return "high"

  elif votes > 500 and votes <2000:
    return "very_high"

  elif votes > 2000:
    return "above_average"

def remove_outliers(df):
    df = df[df["up_votes"] < 1500]

    return df

def word_count_score(df):
    df["word_count"] = df.title.apply(lambda x: len(x.split()))
    return df

def char_count_score(df):
    df["char_count"] = df.title.apply(lambda x: len(x))
    return df

def over_18(df):
    df.over_18 = df.over_18.apply(lambda x: 1 if x else 0)
    return df

def process_df(df):
    labels_scaler = RobustScaler()
    df["up_votes_scaled"] = labels_scaler.fit_transform(df.up_votes.values.reshape(-1,1))
    df_processed = df.drop(["up_votes", "up_votes_scaled", "title", "date_created", "author", "day_created"], axis=1)
    labels = df.up_votes_scaled.values

    return df_processed, labels


def graph_process(df):
    df["votes_category"] = df.up_votes.apply(lambda x: get_category(x))

