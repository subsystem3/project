import re
import string

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import (
    LogisticRegression,
    PassiveAggressiveClassifier,
    RidgeClassifier,
    SGDClassifier,
)

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

import wandb

nltk.download("stopwords")


def preprocess_tweet(tweet):
    # Lowercase the tweet
    tweet = tweet.lower()

    # Remove punctuation
    tweet = tweet.translate(str.maketrans("", "", string.punctuation))

    # Tokenize the tweet
    words = tweet.split()

    # Remove stopwords and stem the words
    stop_words = set(stopwords.words("english"))
    stemmer = SnowballStemmer("english")
    words = [stemmer.stem(word) for word in words if word not in stop_words]

    # Join the words back into a single string
    tweet = " ".join(words)

    return tweet


# Load data
df = pd.read_csv(
    "brands.csv", names=["tweet", "brand", "brand_presence", "sentiment"], header=0
)

# Preprocess tweets
df["tweet"] = df["tweet"].apply(preprocess_tweet)

# Split data
X = df["tweet"]
y_brand_presence = df["brand_presence"]
y_sentiment = df["sentiment"]

(
    X_train_brand_presence,
    X_test_brand_presence,
    y_brand_presence_train,
    y_brand_presence_test,
) = train_test_split(X, y_brand_presence, test_size=0.2)
(
    X_train_sentiment,
    X_test_sentiment,
    y_sentiment_train,
    y_sentiment_test,
) = train_test_split(X, y_sentiment, test_size=0.2)

# Define the pipeline
pipeline = Pipeline(
    [
        ("vectorizer", CountVectorizer()),
        ("scaler", StandardScaler(with_mean=False)),
        ("classifier", LogisticRegression()),
    ]
)

# Define the hyperparameters
hyperparameters = [
    {
        "classifier": [LogisticRegression()],
        "classifier__C": [0.01, 0.1, 1, 10, 100],
        "classifier__max_iter": [5000, 10000],
        "classifier__solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
    },
    {
        "classifier": [MultinomialNB()],
        "classifier__alpha": [0.1, 1, 10, 100],
    },
    {
        "classifier": [BernoulliNB()],
        "classifier__alpha": [0.1, 1, 10, 100],
        "classifier__binarize": [None, 0.0, 0.5, 1.0],
    },
    {
        "classifier": [ComplementNB()],
        "classifier__alpha": [0.1, 1, 10, 100],
        "classifier__norm": [True, False],
    },
    {
        "classifier": [SVC()],
        "classifier__C": [0.1, 1, 10, 100],
        "classifier__kernel": ["linear", "poly", "rbf", "sigmoid"],
    },
    {
        "classifier": [LinearSVC()],
        "classifier__C": [0.1, 1, 10, 100],
    },
    {
        "classifier": [DecisionTreeClassifier()],
        "classifier__max_depth": [5, 10, 20, 50, 100, None],
        "classifier__min_samples_split": [2, 5, 10, 20, 50, 100],
    },
    {
        "classifier": [RandomForestClassifier()],
        "classifier__n_estimators": [10, 50, 100, 200, 500],
        "classifier__max_depth": [5, 10, 20, 50, 100, None],
    },
    {
        "classifier": [AdaBoostClassifier()],
        "classifier__n_estimators": [10, 50, 100, 200, 500],
        "classifier__learning_rate": [0.001, 0.01, 0.1, 1.0],
    },
    {
        "classifier": [GradientBoostingClassifier()],
        "classifier__n_estimators": [10, 50, 100, 200, 500],
        "classifier__learning_rate": [0.001, 0.01, 0.1, 1.0],
    },
    {
        "classifier": [MLPClassifier()],
        "classifier__hidden_layer_sizes": [(10,), (50,), (100,), (200,), (500,)],
        "classifier__activation": ["logistic", "tanh", "relu"],
        "classifier__solver": ["lbfgs", "sgd", "adam"],
        "classifier__learning_rate": ["constant", "invscaling", "adaptive"],
    },
    {
        "classifier": [RidgeClassifier()],
        "classifier__alpha": [0.1, 1, 10, 100],
        "classifier__solver": [
            "auto",
            "svd",
            "cholesky",
            "lsqr",
            "sparse_cg",
            "sag",
            "saga",
        ],
    },
    {
        "classifier": [SGDClassifier()],
        "classifier__loss": [
            "hinge",
            "log",
            "modified_huber",
            "squared_hinge",
            "perceptron",
        ],
        "classifier__penalty": ["l1", "l2", "elasticnet"],
        "classifier__alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
    },
    {
        "classifier": [PassiveAggressiveClassifier()],
        "classifier__C": [0.1, 1, 10, 100],
        "classifier__fit_intercept": [True, False],
        "classifier__early_stopping": [True, False],
    },
    {
        "classifier": [KNeighborsClassifier()],
        "classifier__n_neighbors": [1, 5, 10, 25, 50],
        "classifier__weights": ["uniform", "distance"],
    },
]

# Grid search with cross-validation
for params in hyperparameters:
    classifier_name = params["classifier"][0].__class__.__name__

    # Initialize wandb
    wandb.init(project="brand-sentiment", name=classifier_name + "_brand")

    grid_search_brand_presence = GridSearchCV(
        pipeline, params, cv=5, scoring="accuracy"
    )
    grid_search_brand_presence.fit(X_train_brand_presence, y_brand_presence_train)

    # Log the best parameters
    wandb.log(grid_search_brand_presence.best_params_)

    # Evaluate the model
    y_brand_presence_pred = grid_search_brand_presence.predict(X_test_brand_presence)
    accuracy_brand_presence = accuracy_score(
        y_brand_presence_test, y_brand_presence_pred
    )
    precision_brand_presence = precision_score(
        y_brand_presence_test, y_brand_presence_pred
    )
    recall_brand_presence = recall_score(y_brand_presence_test, y_brand_presence_pred)
    f1_brand_presence = f1_score(y_brand_presence_test, y_brand_presence_pred)
    confusion_brand_presence = confusion_matrix(
        y_brand_presence_test, y_brand_presence_pred
    )
    report_brand_presence = classification_report(
        y_brand_presence_test, y_brand_presence_pred
    )

    # Log the metrics
    wandb.log(
        {
            "accuracy_brand_presence": accuracy_brand_presence,
            "precision_brand_presence": precision_brand_presence,
            "recall_brand_presence": recall_brand_presence,
            "f1_brand_presence": f1_brand_presence,
            "confusion_matrix_brand_presence": confusion_brand_presence,
            "classification_report_brand_presence": report_brand_presence,
        }
    )

    # Finish the wandb run
    wandb.finish()

    # Repeat the same process for sentiment
    wandb.init(project="brand-sentiment", name=classifier_name + "_sentiment")
    grid_search_sentiment = GridSearchCV(pipeline, params, cv=5, scoring="accuracy")
    grid_search_sentiment.fit(X_train_sentiment, y_sentiment_train)
    wandb.log(grid_search_sentiment.best_params_)
    y_sentiment_pred = grid_search_sentiment.predict(X_test_sentiment)
    accuracy_sentiment = accuracy_score(y_sentiment_test, y_sentiment_pred)
    precision_sentiment = precision_score(y_sentiment_test, y_sentiment_pred)
    recall_sentiment = recall_score(y_sentiment_test, y_sentiment_pred)
    f1_sentiment = f1_score(y_sentiment_test, y_sentiment_pred)
    confusion_sentiment = confusion_matrix(y_sentiment_test, y_sentiment_pred)
    report_sentiment = classification_report(y_sentiment_test, y_sentiment_pred)
    wandb.log(
        {
            "accuracy_sentiment": accuracy_sentiment,
            "precision_sentiment": precision_sentiment,
            "recall_sentiment": recall_sentiment,
            "f1_sentiment": f1_sentiment,
            "confusion_matrix_sentiment": confusion_sentiment,
            "classification_report_sentiment": report_sentiment,
        }
    )
    wandb.finish()
