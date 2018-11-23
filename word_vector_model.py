from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import numpy as np
import csv
import os

def clean(posts):

    # Cleaning preparation
    re_pattern = '[a-z]{3,}'  # only words with length 3 or more
    stopwords_set = set(stopwords.words('english'))     # get stopwords
    tokenizer = RegexpTokenizer(re_pattern)             # build tokenizer
    stemmer = PorterStemmer()

    # Cleaning steps:
    # Convert to lower case
    # Tokenize
    # Remove Stopwords
    # Stem words

    cleaned_docs = list()
    for p in posts:
        text = p.lower()
        tokens = tokenizer.tokenize(text)   # only keep words with length 3 or more
        tokens_wo_stopwords = [w for w in tokens if w not in stopwords_set]
        doc_tokens = [stemmer.stem(w) for w in tokens_wo_stopwords] # stemming
        if len(doc_tokens) > 3:
            cleaned_docs.append(doc_tokens)

    return cleaned_docs

def read_file(fname):

    with open(fname, newline='\n') as csvfile:
        datasets = csv.reader(csvfile, delimiter=',')

        threads = list()    # list of list
        for row in datasets:
            threads.append(row)

        # convert to ndarray
        threads = np.asarray(threads)

        # get rid of the unnecessary columns
        threads = threads[2:,:] # remove the header lines
        thread_upvotes = np.where(threads[:,8] > 0)     # remove posts with negative upvotes
        thread_titles = np.take(threads[:, 2], thread_upvotes)  # X
        subreddit_labels = np.take(threads[:, 4], thread_upvotes)   # y

        X = clean(thread_titles)
        y = subreddit_labels

    return X, y

def get_dataset_splits(X, y):

    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        X = np.asarray(X)
        y = np.asarray(y)

    # seed for random number generator
    seed = 42

    data_splits = dict()

    # split into train, test sets  [70:30]
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.3,
                                                        random_state=seed,
                                                        shuffle=False)

    data_splits['X_train'] = X_train
    data_splits['y_train'] = y_train

    data_splits['X_test'] = X_test
    data_splits['y_test'] = y_test

    return data_splits

def get_model(documents):

    # build a word2vec model and vocabulary
    model = Word2Vec(documents,
                     size=100,  # vector size
                     window=5,  # context window
                     min_count=10,
                     workers=4)

    # train model
    model.train(documents)

    return model

def get_document_vectors(wv_model, X):
    max_len = 0
    features = list()
    for post in X:
        n = len(post)
        sum = np.zeros(n)
        for word in post:
            sum += wv_model.wv[word]

        # compute an average of all words in a post
        avg = sum / n
        max_len = max( max_len, len(avg) )
        features.append(avg)

    return features

if __name__ == '__main__':

    dataset_dir = 'dataset'
    subreddits_fname = ['entertainment_anime.csv', 'entertainment_comicbooks.csv', 'entertainment_harrypotter.csv',
                        'entertainment_movies.csv', 'entertainment_music.csv', 'entertainment_starwars.csv']

    test_sr = subreddits_fname[0]
    fpath = os.path.join(dataset_dir, test_sr)
    X, y = read_file(fpath)
    splits = get_dataset_splits(X, y)
    wv_model = get_model(X)
    doc_vectors = get_document_vectors(wv_model, X)

    # for sr in subreddits_fname:
    #     fpath = os.path.join(dataset_dir, sr)
    #     X, y = read_file(fpath)
    #     splits = get_dataset_splits(X, y)
    #     wv_model = get_model(X)
    #     doc_vectors = get_document_vectors(wv_model, X)
    #
    #





