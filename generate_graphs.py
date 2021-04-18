import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
import string
nltk.download('stopwords')
nltk.download('punkt')

stopwords = set(stopwords.words('english'))

best_titles = df.sort_values(by='up_votes', ascending=False)['title'].values[:10]
best_words = set(np.concatenate([word_tokenize(t) for t in best_titles])) - stopwords
best_words = {word.lower() for word in best_words}
best_words = best_words - set(string.punctuation) - set(string.digits)

def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

best_words = {word for word in best_words if not is_float(word)}

best_words = {word for word in best_words if "'" not in word}  # drop contractions


words_tokenized = [[w.lower() for w in word_tokenize(t)] for t in df['title']]
df['tokenized_title'] = words_tokenized
all_words = pd.Series(np.concatenate(words_tokenized)).value_counts()

all_words = all_words[[word not in stopwords for word in all_words.index]]
all_words = all_words[[word not in string.punctuation for word in all_words.index]]
all_words = all_words[[word not in string.digits for word in all_words.index]]
all_words = all_words[[not is_float(word) for word in all_words.index]]
all_words = all_words[["'" not in word for word in all_words.index]]
plt.figure(figsize=(10,10))
sns.barplot(x =all_words[:30].values, y=all_words[:30].index)


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

df["votes_category"] = df.up_votes.apply(lambda x: get_category(x))

plt.figure(figsize=(10,10))
df['votes_category'].value_counts().plot(kind="bar")

