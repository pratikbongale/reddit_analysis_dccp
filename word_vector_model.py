from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import csv
import os

def clean(posts, labels):
    '''

    :param posts: list of strings
    :param labels: list of strings
    :return: list of lists(tokens)
    '''


    # Cleaning preparation
    re_pattern = '[a-z]{3,}'  # only words with length 3 or more
    stopwords_set = set(stopwords.words('english'))     # get stopwords
    tokenizer = RegexpTokenizer(re_pattern)             # build tokenizer
    stemmer = SnowballStemmer("english")

    # Cleaning steps:
    # Convert to lower case
    # Tokenize
    # Remove Stopwords
    # Stem words

    cleaned_docs = list()
    deleted_doc_idx = list()
    for i, p in enumerate(posts):
        p = p.strip()
        text = p.lower()
        tokens = tokenizer.tokenize(text)   # only keep words with length 3 or more
        tokens_wo_stopwords = [w for w in tokens if w not in stopwords_set]
        doc_tokens = [stemmer.stem(w) for w in tokens_wo_stopwords] # stemming
        if len(doc_tokens) > 3:
            cleaned_docs.append(doc_tokens)
        else:
            deleted_doc_idx.append(i)

    arr = np.array(labels)
    arr = np.delete(arr, deleted_doc_idx, axis=0)
    cleaned_labels = arr.tolist()

    return cleaned_docs, cleaned_labels

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

def get_wv_model(sentences):

    # build a word2vec model
    model = Word2Vec(sentences,
                     size=100,  # vector size
                     window=5,  # context window
                     min_count=1,
                     workers=4)

    return model

def get_document_vectors(wv_model, X):
    features = list()
    for post_tokens in X:
        n = wv_model.vector_size
        sum = np.zeros(n)
        for word in post_tokens:
            sum += wv_model.wv[word]

        # compute an average of all words in a post
        avg = sum / len(post_tokens)
        avg = avg.tolist()
        features.append(avg)

    return features

if __name__ == '__main__':

    dataset_dir = 'dataset'
    subreddits_fname = ['entertainment_anime.csv', 'entertainment_comicbooks.csv', 'entertainment_harrypotter.csv',
                        'entertainment_movies.csv']

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
    X_train = splits['X_train']
    y_train = splits['y_train']
    X_train, y_train = clean(X_train, y_train)

    X_test = splits['X_test']
    y_test = splits['y_test']
    X_test, y_test = clean(X_test, y_test)

    # train the word vector model on entire dataset
    corpus = X_train + X_test
    wv_model = get_wv_model(corpus)   # needs the entire corpus to create vocabulary
    wv_model.train(corpus, total_examples=len(corpus), epochs=wv_model.iter)
    X_train = get_document_vectors(wv_model, X_train)

    # train a multi-class classifier model using only the training data
    clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=600, learning_rate=0.5)
    clf.fit(X_train, y_train)

    # get document vectors using word vector model
    X_test = get_document_vectors(wv_model, X_test)

    # get predictions from the trained classifier model
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print('Accuracy:', accuracy)