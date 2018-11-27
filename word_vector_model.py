from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import csv
import os

def read_file(fname):

    with open(fname, newline='\n') as csvfile:
        datasets = csv.reader(csvfile, delimiter=',')

        threads = list()    # list of list
        for i, row in enumerate(datasets):
            threads.append(row)
            if i > 5000:
                break

        # convert to ndarray
        threads = np.array(threads)

        if len(threads[0]) == 12:
            post_col = 1
            upvotes_col = 7
            subr_col = 3
        else:
            post_col = 2
            upvotes_col = 8
            subr_col = 4

        # get rid of the unnecessary rows and column
        threads = threads[2:,:] # remove the header lines

        # get rid of empty posts
        threads = threads[threads[:,post_col] != '']

        upvotes = threads[:,upvotes_col].astype(float)
        thread_upvotes = np.where(upvotes > 0.0)     # remove posts with negative upvotes
        thread_titles = np.take(threads[:, post_col], thread_upvotes)  # X
        subreddit_labels = np.take(threads[:, subr_col], thread_upvotes)   # y

        X = thread_titles
        y = subreddit_labels

    return X, y

def clean(posts, labels):
    '''

    :param posts: list of strings
    :param labels: list of strings
    :return: list of lists(tokens)
    '''

    # Cleaning preparation
    re_pattern = '[a-z]{2,}'  # only words with length 2 or more
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
        tokens = [w for w in tokens if w not in stopwords_set]
        tokens = [stemmer.stem(w) for w in tokens] # stemming
        if len(tokens) > 4:     # we need atleast 4 words in a post
            cleaned_docs.append(tokens)
        else:
            deleted_doc_idx.append(i)

    arr = np.array(labels)
    arr = np.delete(arr, deleted_doc_idx, axis=0)
    cleaned_labels = arr.tolist()

    return cleaned_docs, cleaned_labels

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

def get_bow_features(X_train, X_test):

    #
    # Train features
    #
    inp = list()
    for s in X_train:
        inp.append(' '.join(s))

    count_vect = CountVectorizer(min_df=10,
                                 encoding='latin-1',
                                 ngram_range=(1, 3),
                                 stop_words='english')

    X_train_counts = count_vect.fit_transform(inp)

    tfidf_transformer = TfidfTransformer(sublinear_tf=True,norm='l2')

    X_train = tfidf_transformer.fit_transform(X_train_counts)


    #
    # Test features
    #
    inp = list()
    for s in X_test:
        inp.append(' '.join(s))

    X_test = count_vect.transform(inp)

    return X_train, X_test

def get_wv_features(X_train, X_test):

    corpus = X_train + X_test
    wv_model = get_wv_model(corpus)  # needs the entire corpus to create vocabulary
    wv_model.train(corpus, total_examples=len(corpus), epochs=wv_model.iter)
    X_train = get_document_vectors(wv_model, X_train)
    X_test = get_document_vectors(wv_model, X_test)

    return X_train, X_test

def get_wv_model(sentences):

    # build a word2vec model
    model = Word2Vec(sentences,
                     size=200,  # vector size
                     window=5,  # context window
                     min_count=10, # count in corpus
                     workers=4)

    return model

def get_document_vectors(wv_model, X):
    features = list()
    for post_tokens in X:
        n = wv_model.vector_size
        sum = np.zeros(n)
        wc = 0.0
        for word in post_tokens:
            if word in wv_model.wv:
                sum += wv_model.wv[word]
                wc += 1.0

        # compute an average of all words in a post
        if wc > 0.0:
            avg = sum / wc
        else:
            avg = sum
        avg = avg.tolist()
        features.append(avg)

    return features

def evaluate_adaboost_model(X_train, X_test):

    # train a multi-class classifier model using only the training data
    clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=600, learning_rate=0.5)
    clf.fit(X_train, y_train)

    # get predictions from the trained classifier model
    y_pred = clf.predict(X_test)

    return y_pred

def evaluate_nb_model(X_train, X_test):

    # train a multi-class classifier model using only the training data
    clf = MultinomialNB()
    clf.fit(X_train, y_train)

    # get predictions from the trained classifier model
    y_pred = clf.predict(X_test)

    return y_pred

def evaluate_svc_model(X_train, X_test):

    # train a multi-class classifier model using only the training data
    clf = SVC()
    clf.fit(X_train, y_train)

    # get predictions from the trained classifier model
    y_pred = clf.predict(X_test)

    return y_pred

def check_classify_types(X_train, Y_train, X_test, Y_test, list_classifiers, features, verbose=True):

    dict_models = {}
    for classifier_name, classifier in list(list_classifiers.items())[:8]:
        classifier.fit(X_train, Y_train)

        Y_pred = classifier.predict(X_test)
        plt.title(classifier_name)
        plot_confusion_matrix(Y_test, Y_pred)

        fig_name = '_'.join([features, classifier_name, '.png'])
        plt.savefig('Results/' + fig_name)

        train_score = classifier.score(X_train, Y_train)
        test_score = classifier.score(X_test, Y_test)

        dict_models[classifier_name] = {'model': classifier, 'train_score': train_score, 'test_score': test_score,}

    return dict_models


def display_models(dict_models, sort_by='test_score'):
    cls = [key for key in dict_models.keys()]
    test_s = [dict_models[key]['test_score'] for key in cls]
    training_s = [dict_models[key]['train_score'] for key in cls]

    df_ = pd.DataFrame(data=np.zeros(shape=(len(cls), 3)),
                       columns=['classifier', 'train_score', 'test_score'])
    for ii in range(0, len(cls)):
        df_.loc[ii, 'classifier'] = cls[ii]
        df_.loc[ii, 'train_score'] = training_s[ii]
        df_.loc[ii, 'test_score'] = test_s[ii]

    print(df_.sort_values(by=sort_by, ascending=False))

def plot_confusion_matrix(y_test, y_pred):
    conf_mat = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(10, 10))

    labels = set(y_test)
    sns.heatmap(conf_mat, annot=True, fmt='d',
                xticklabels=labels, yticklabels=labels)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    # plt.show()

if __name__ == '__main__':

    dataset_dir = 'dataset'

    subreddits_fname_phase1 = ['entertainment_music.csv', 'gaming_gaming.csv', 'learning_science.csv',
                               'lifestyle_food.csv', 'news_politics.csv']

    subreddits_fname_phase2 = ['entertainment_anime.csv', 'entertainment_comicbooks.csv',
                               'entertainment_harrypotter.csv',
                               'entertainment_starwars.csv', 'entertainment_music.csv']

    # build dataset with equal priors
    X = list()
    y = list()
    for sr in subreddits_fname_phase2:
        fpath = os.path.join(dataset_dir, sr)
        threads, labels = read_file(fpath)
        t = threads[0].tolist()
        l = labels[0].tolist()
        X.extend(t)
        y.extend(l)

    # split dataset (train/test) preserving the percentage of samples for each class
    splits = get_dataset_splits(X, y)

    # clean and tokenize the dataset
    X_train = splits['X_train']
    y_train = splits['y_train']
    X_train, y_train = clean(X_train, y_train)

    X_test = splits['X_test']
    y_test = splits['y_test']
    X_test, y_test = clean(X_test, y_test)

    X_train_bow, X_test_bow = get_bow_features(X_train, X_test)
    X_train_wv, X_test_wv = get_wv_features(X_train, X_test)

    list_classifiers = {
        "Logistic Regression": LogisticRegression(),
        "Nearest Neighbors": KNeighborsClassifier(),
        "Linear SVM": SVC(),
        "Gradient Boosting Classifier": GradientBoostingClassifier(n_estimators=1000),
        "Decision Tree": DecisionTreeClassifier(max_depth=100),
        "Random Forest": RandomForestClassifier(n_estimators=1000),
        "Naive Bayes Multinomial": MultinomialNB(),
        "Adaboost": AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=600, learning_rate=0.5)
    }

    print('='*30)
    print('Bag of words')
    print('=' * 30)
    dict_models = check_classify_types(X_train_bow, y_train, X_test_bow, y_test, list_classifiers, 'bow')
    display_models(dict_models)

    del list_classifiers["Naive Bayes Multinomial"]
    print('=' * 30)
    print('Word vectors')
    print('=' * 30)
    dict_models = check_classify_types(X_train_wv, y_train, X_test_wv, y_test, list_classifiers, 'wv')
    display_models(dict_models)




