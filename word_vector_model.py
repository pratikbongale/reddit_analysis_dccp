from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.cross_validation import StratifiedKFold
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
    for p in posts.T:
        p = p[0]
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
        for i, row in enumerate(datasets):
            threads.append(row)
            if i > 200:
                break

        # convert to ndarray
        threads = np.array(threads)

        # get rid of the unnecessary rows and column
        threads = threads[2:,:] # remove the header lines

        # get rid of empty posts
        threads = threads[threads[:,2] != '']

        upvotes = threads[:,8].astype(float)
        thread_upvotes = np.where(upvotes > 0.0)     # remove posts with negative upvotes
        thread_titles = np.take(threads[:, 2], thread_upvotes)  # X
        subreddit_labels = np.take(threads[:, 4], thread_upvotes)   # y

        X = thread_titles
        y = subreddit_labels

    return X, y

def get_dataset_splits(X, y):

    # seed for random number generator
    seed = 42

    data_splits = dict()

    # split into train, test sets  [70:30]
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.3,
                                                        random_state=seed,
                                                        shuffle=True,
                                                        stratify=y)

    data_splits['X_train'] = X_train
    data_splits['y_train'] = y_train

    data_splits['X_test'] = X_test
    data_splits['y_test'] = y_test

    return data_splits

def get_model(sentences):

    # build a word2vec model
    model = Word2Vec(sentences,
                     size=100,  # vector size
                     window=5,  # context window
                     min_count=1,
                     workers=4)

    # # build vocabulary
    # model.build_vocab(sentences)

    # train model
    model.train(sentences, total_examples=len(sentences), epochs=model.iter)

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
                        'entertainment_movies.csv']

    # test_sr = subreddits_fname[0]
    # fpath = os.path.join(dataset_dir, test_sr)
    # X, y = read_file(fpath)     # returns documents with their corresponding labels
    #
    # wv_model = get_model(X)     # wv dimension : 100
    # doc_vectors = get_document_vectors(wv_model, X)
    # splits = get_dataset_splits(doc_vectors, y)




    # build dataset with equal priors
    X = list()
    y = list()
    for sr in subreddits_fname:
        fpath = os.path.join(dataset_dir, sr)
        threads, labels = read_file(fpath)
        t = threads[0].tolist()
        l = labels[0].tolist()
        X.extend(t)
        y.extend(l)

    print('Complete dataset', len(X))

    # split dataset (train/test) preserving the percentage of samples for each class
    splits = get_dataset_splits(X, y)

    # clean and tokenize the dataset


    # train a word2vec model using only the training data


    # test model on test data






