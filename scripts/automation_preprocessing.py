import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.base import BaseEstimator, TransformerMixin

class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, extra_stopwords=None):
        self.stop_words = set(stopwords.words('english'))
        if extra_stopwords:
            self.stop_words.update(extra_stopwords)
        self.stemmer = PorterStemmer()

        # Slang dictionary
        self.slangwords = {
            "u": "you", "ur": "your", "gonna": "going to",
            "wanna": "want to", "gotta": "got to", "lol": "laugh",
            "brb": "be right back", "thx": "thanks", "omg": "oh my god",
            "idk": "i don't know", "btw": "by the way", "imo": "in my opinion",
            "pls": "please", "smh": "shaking my head"
        }

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        processed = X.apply(self._preprocess)
        return processed.fillna("empty").replace(r'^\s*$', "empty", regex=True)

    def _preprocess(self, text):
        if pd.isnull(text) or not str(text).strip():
            return "empty"

        # Lowercase
        text = text.lower()

        # Cleaning text
        text = re.sub(r'@[A-Za-z0-9_]+', '', text)  # mentions
        text = re.sub(r'#[A-Za-z0-9_]+', '', text)  # hashtags
        text = re.sub(r"http\S+|www\S+", '', text)  # URLs
        text = re.sub(r"[^a-zA-Z\s]", '', text)     # punctuation & numbers

        # Tokenization
        tokens = word_tokenize(text)

        # Slang normalization
        tokens = [self.slangwords.get(word, word) for word in tokens]

        # Stopword removal
        tokens = [word for word in tokens if word not in self.stop_words]

        # Stemming
        stemmed = [self.stemmer.stem(word) for word in tokens]

        # Join back to string
        return ' '.join(stemmed)
