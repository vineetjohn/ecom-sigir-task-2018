from nltk.corpus import stopwords
from sklearn.feature_extraction import stop_words
from spacy.lang.en.stop_words import STOP_WORDS as spacy_stopwords

nltk_stopwords = set(stopwords.words('english'))
sklearn_stopwords = stop_words.ENGLISH_STOP_WORDS

custom_stopwords = {'mmx'}
custom_stopwords = custom_stopwords.union(spacy_stopwords)
custom_stopwords = custom_stopwords.union(nltk_stopwords)
custom_stopwords = custom_stopwords.union(sklearn_stopwords)
