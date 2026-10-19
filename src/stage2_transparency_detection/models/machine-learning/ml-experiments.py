from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
import tensorflow_hub as thub
from scipy.sparse import hstack
from numpy import hstack as nphstack
from gensim.models import KeyedVectors
from tqdm import tqdm
import gensim.downloader as api
import pickle


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ========= CONFIG =========

DATA_PATH = Path("data/ground_truth/Cleaned-Ground-Truth-DL.xlsx")

# Thresholds for opt_tresh
THRESHOLDS = [
    0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49,
    0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59,
    0.6, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69,
]


# ========= DATA LOADING =========

pd_data = pd.read_excel(DATA_PATH)
x_data = np.array(pd_data["clean_review"].tolist())
y_data = np.array(pd_data["Label"].astype(int).to_list())


# ========= HELPERS =========

def transform_X(vectorizer, x_train, x_test):
    x_train_B = vectorizer.fit_transform(x_train)
    x_test_B = vectorizer.transform(x_test)
    return x_train_B, x_test_B


def fit_predict(classifier, x_train, x_test, y_train):
    classifier.fit(x_train, y_train)
    model = classifier
    predictions = classifier.predict_proba(x_test)
    return predictions, model


def scores(y_test, predictions):
    precision = metrics.precision_score(y_test, predictions)
    recall = metrics.recall_score(y_test, predictions)
    f1 = metrics.f1_score(y_test, predictions)
    acc = metrics.accuracy_score(y_test, predictions)
    return precision, recall, f1, acc


def plot_seaborn(Y_test, Y_pred, name):
    class_names = ["Non-Fairness", "Fairness"]
    cm = confusion_matrix(Y_test, Y_pred)
    import seaborn as sn
    plt.figure(figsize=(6, 5))
    sns = sn.heatmap(cm, annot=True, xticklabels=class_names, yticklabels=class_names)
    figure = sns.get_figure()
    # figure.savefig(f"{name}.png", dpi=400)


def saving_model(classifiermodel, modelname):
    pkl_filename = f"Model-{modelname}.pkl"
    with open(pkl_filename, "wb") as file:
        pickle.dump(classifiermodel, file)


def loading_model(filepath_savedmodel):
    with open(filepath_savedmodel, "rb") as file:
        clf = pickle.load(file)
    return clf


def tfidf_char(x_train, x_test):
    vectorizer = TfidfVectorizer(
        max_features=50000,
        strip_accents="unicode",
        analyzer="char",
        sublinear_tf=False,
        ngram_range=(4, 6),
    )
    x_train_B, x_test_B = transform_X(vectorizer, x_train, x_test)
    return x_train_B, x_test_B


def opt_tresh(y_test, predictions):
    max_precision = 0
    best_treshold = 0
    y_pred = []

    for tresh in THRESHOLDS:
        pred = []
        for row in predictions:
            if row[0] > tresh:
                pred.append(0)
            else:
                pred.append(1)
        precision = metrics.precision_score(y_test, pred)
        if precision > max_precision:
            max_precision = precision
            best_treshold = tresh

    print("best tresh is: ", best_treshold)

    for row in predictions:
        if row[0] > best_treshold:
            y_pred.append(0)
        else:
            y_pred.append(1)

    return y_pred


# ========= USE (Universal Sentence Encoder) =========

embed = thub.load("https://tfhub.dev/google/universal-sentence-encoder/4")


def USEconvert(x_train, x_test):
    x_train_B = embed(x_train)
    x_test_B = embed(x_test)
    return x_train_B, x_test_B


# ========= Word2Vec =========

word2vec_model = api.load("word2vec-google-news-300")
word2vec_words = list(word2vec_model.key_to_index)


def vectorize_w2v(dataset, w2v_model, w2v_words):
    word2vec_corpus = [sentence.split() for sentence in dataset]

    sent_vectors = []
    for sentence in tqdm(word2vec_corpus):
        sent_vec = np.zeros(300)
        count_words = 0
        for word in sentence:
            if word in w2v_words:
                word_vectors = w2v_model[word]
                sent_vec += word_vectors
                count_words += 1
        if count_words != 0:
            sent_vec /= count_words
        sent_vectors.append(sent_vec)

    return np.array(sent_vectors)


X_w2v_all = vectorize_w2v(x_data, word2vec_model, word2vec_words)


# ========= Combination functions =========

def tfidf_USE(x_train, x_test):
    x_train_tc, x_test_tc = tfidf_char(x_train, x_test)
    x_train_u, x_test_u = USEconvert(x_train, x_test)
    xtrain_comb = hstack((x_train_tc, x_train_u))
    xtest_comb = hstack((x_test_tc, x_test_u))
    return xtrain_comb, xtest_comb


def tfidf_wv(x_train, x_test):
    x_train_tc, x_test_tc = tfidf_char(x_train, x_test)
    x_train_wv = vectorize_w2v(x_train, word2vec_model, word2vec_words)
    x_test_wv = vectorize_w2v(x_test, word2vec_model, word2vec_words)
    xtrain_comb = hstack((x_train_tc, x_train_wv))
    xtest_comb = hstack((x_test_tc, x_test_wv))
    return xtrain_comb, xtest_comb


def use_wv(x_train, x_test):
    x_train_u, x_test_u = USEconvert(x_train, x_test)
    x_train_wv = vectorize_w2v(x_train, word2vec_model, word2vec_words)
    x_test_wv = vectorize_w2v(x_test, word2vec_model, word2vec_words)
    xtrain_comb = nphstack((x_train_wv, x_train_u))
    xtest_comb = nphstack((x_test_wv, x_test_u))
    return xtrain_comb, xtest_comb


def tfidf_use_wv(x_train, x_test):
    x_train_tc, x_test_tc = tfidf_char(x_train, x_test)
    x_train_u, x_test_u = USEconvert(x_train, x_test)
    x_train_wv = vectorize_w2v(x_train, word2vec_model, word2vec_words)
    x_test_wv = vectorize_w2v(x_test, word2vec_model, word2vec_words)
    xtrain_comb = hstack((x_train_tc, x_train_u, x_train_wv))
    xtest_comb = hstack((x_test_tc, x_test_u, x_test_wv))
    return xtrain_comb, xtest_comb


# ========= Generic K-fold runner =========

def run_kfold(
    modelname,
    x_data,
    y_data,
    feature_func,
    classifier_factory,
    k_folds=10,
    save_best_model=True,
):
    print(f"\nTraining K-fold validation for {modelname} model !!!!!")
    kfold = KFold(n_splits=k_folds, shuffle=False)

    max_preci = 0
    l_f1score, l_acc, l_precision, l_recall = [], [], [], []
    l_ytest, l_ypred, l_y_proba = [], [], []

    bestmodel = None
    bestfold = None

    for fold, (train_ids, test_ids) in enumerate(kfold.split(x_data)):
        x_train = [x_data[t] for t in train_ids]
        y_train = [y_data[t] for t in train_ids]
        x_test = [x_data[t] for t in test_ids]
        y_test = [y_data[t] for t in test_ids]

        print(f"FOLD {fold + 1} ...")

        x_train_B, x_test_B = feature_func(x_train, x_test)

        classifier = classifier_factory()
        predictions, classifiermodel = fit_predict(classifier, x_train_B, x_test_B, y_train)
        y_pred = opt_tresh(y_test, predictions)

        precisionscore, recallscore, f1score, accscore = scores(y_test, y_pred)

        print("precision:", precisionscore)
        print("recall:", recallscore)
        print("f1:", f1score)
        print("acc:", accscore)

        if precisionscore > max_preci:
            max_preci = precisionscore
            bestmodel = classifiermodel
            bestfold = fold + 1

        l_ytest.extend(y_test)
        l_ypred.extend(y_pred)
        l_y_proba.extend(classifiermodel.predict_proba(x_test_B)[:, 1])

        l_f1score.append(f1score)
        l_acc.append(accscore)
        l_precision.append(precisionscore)
        l_recall.append(recallscore)

    if save_best_model and bestmodel is not None:
        print(f"\n\nBest model is in fold {bestfold}!!!\n Best Model saved!!!")
        saving_model(bestmodel, modelname)

    print(f"\n\nK-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS")
    print("--------------------------------")

    sum_f1score = sum(l_f1score)
    sum_acc = sum(l_acc)
    sum_precision = sum(l_precision)
    sum_recall = sum(l_recall)

    print(f"Precision: {sum_precision / len(l_precision)}")
    print(f"Recall: {sum_recall / len(l_recall)}")
    print(f"F1-score: {sum_f1score / len(l_f1score)}")
    print(f"Accuracy: {sum_acc / len(l_acc)}")

    return {
        "y_test": np.array(l_ytest),
        "y_pred": np.array(l_ypred),
        "y_proba": np.array(l_y_proba),
        "precision": sum_precision / len(l_precision),
        "recall": sum_recall / len(l_recall),
        "f1": sum_f1score / len(l_f1score),
        "acc": sum_acc / len(l_acc),
    }


# ========= Main: run all experiments =========

def main():
    # TFIDF (char)
    run_kfold(
        modelname="LogisticRegression-char",
        x_data=x_data,
        y_data=y_data,
        feature_func=tfidf_char,
        classifier_factory=lambda: LogisticRegression(penalty="l2", solver="liblinear"),
    )
    run_kfold(
        modelname="SVM-linear-char",
        x_data=x_data,
        y_data=y_data,
        feature_func=tfidf_char,
        classifier_factory=lambda: SVC(kernel="linear", probability=True),
    )
    run_kfold(
        modelname="RF-char",
        x_data=x_data,
        y_data=y_data,
        feature_func=tfidf_char,
        classifier_factory=lambda: RandomForestClassifier(criterion="gini", n_estimators=100),
    )

    # USE
    run_kfold(
        modelname="LogisticRegression-USE",
        x_data=x_data,
        y_data=y_data,
        feature_func=USEconvert,
        classifier_factory=lambda: LogisticRegression(penalty="l2", solver="liblinear"),
    )
    run_kfold(
        modelname="SVM-linear-USE",
        x_data=x_data,
        y_data=y_data,
        feature_func=USEconvert,
        classifier_factory=lambda: SVC(kernel="linear", probability=True),
    )
    run_kfold(
        modelname="RF-USE",
        x_data=x_data,
        y_data=y_data,
        feature_func=USEconvert,
        classifier_factory=lambda: RandomForestClassifier(criterion="gini"),
    )

    # Word2Vec (precomputed X_w2v_all)
    run_kfold(
        modelname="LogisticRegression-w2v",
        x_data=X_w2v_all,
        y_data=y_data,
        feature_func=lambda x_tr, x_te: (x_tr, x_te),
        classifier_factory=lambda: LogisticRegression(penalty="l2", solver="liblinear"),
    )
    run_kfold(
        modelname="SVM-linear-w2v",
        x_data=X_w2v_all,
        y_data=y_data,
        feature_func=lambda x_tr, x_te: (x_tr, x_te),
        classifier_factory=lambda: SVC(kernel="linear", probability=True),
    )
    run_kfold(
        modelname="RF-w2v",
        x_data=X_w2v_all,
        y_data=y_data,
        feature_func=lambda x_tr, x_te: (x_tr, x_te),
        classifier_factory=lambda: RandomForestClassifier(criterion="gini"),
    )

    # TFIDF + USE
    run_kfold(
        modelname="LR_TU",
        x_data=x_data,
        y_data=y_data,
        feature_func=tfidf_USE,
        classifier_factory=lambda: LogisticRegression(penalty="l2", solver="liblinear"),
    )
    run_kfold(
        modelname="SVM-linear-TU",
        x_data=x_data,
        y_data=y_data,
        feature_func=tfidf_USE,
        classifier_factory=lambda: SVC(kernel="linear", probability=True),
    )
    run_kfold(
        modelname="RF-TU",
        x_data=x_data,
        y_data=y_data,
        feature_func=tfidf_USE,
        classifier_factory=lambda: RandomForestClassifier(criterion="gini"),
    )

    # TFIDF + W2V
    run_kfold(
        modelname="LR_TW",
        x_data=x_data,
        y_data=y_data,
        feature_func=tfidf_wv,
        classifier_factory=lambda: LogisticRegression(penalty="l2", solver="liblinear"),
    )
    run_kfold(
        modelname="SVM-linear-TW",
        x_data=x_data,
        y_data=y_data,
        feature_func=tfidf_wv,
        classifier_factory=lambda: SVC(kernel="linear", probability=True),
    )
    run_kfold(
        modelname="RF-TW",
        x_data=x_data,
        y_data=y_data,
        feature_func=tfidf_wv,
        classifier_factory=lambda: RandomForestClassifier(criterion="gini"),
    )

    # USE + W2V
    run_kfold(
        modelname="LR_UW",
        x_data=x_data,
        y_data=y_data,
        feature_func=use_wv,
        classifier_factory=lambda: LogisticRegression(penalty="l2", solver="liblinear"),
    )
    run_kfold(
        modelname="SVM-linear-UW",
        x_data=x_data,
        y_data=y_data,
        feature_func=use_wv,
        classifier_factory=lambda: SVC(kernel="linear", probability=True),
    )
    run_kfold(
        modelname="RF-UW",
        x_data=x_data,
        y_data=y_data,
        feature_func=use_wv,
        classifier_factory=lambda: RandomForestClassifier(criterion="gini"),
    )

    # TFIDF + USE + W2V
    run_kfold(
        modelname="LR_TUW",
        x_data=x_data,
        y_data=y_data,
        feature_func=tfidf_use_wv,
        classifier_factory=lambda: LogisticRegression(penalty="l2", solver="liblinear"),
    )
    run_kfold(
        modelname="SVM-linear-TUW",
        x_data=x_data,
        y_data=y_data,
        feature_func=tfidf_use_wv,
        classifier_factory=lambda: SVC(kernel="linear", probability=True),
    )
    run_kfold(
        modelname="RF-TUW",
        x_data=x_data,
        y_data=y_data,
        feature_func=tfidf_use_wv,
        classifier_factory=lambda: RandomForestClassifier(criterion="gini"),
    )


if __name__ == "__main__":
    main()