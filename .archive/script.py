import os

import dotenv
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

import wandb

dotenv.load_dotenv()
wandb.finish()
wandb.login(key=os.getenv("WANDB_API_KEY"))
pd.set_option("display.max_colwidth", 20)

FIELDS = ["target", "ids", "date", "flag", "user", "text"]
CLASS_NAMES = ["Negative", "Neutral", "Positive"]

df_train = pd.read_csv(
    "datasets/training.1600000.processed.noemoticon.csv",
    encoding="ISO-8859-1",
    names=FIELDS,
)
df_train["target"] = df_train["target"].map({0: 0, 2: 1, 4: 2})
df_train.head()

df_test = pd.read_csv(
    "datasets/testdata.manual.2009.06.14.csv",
    encoding="ISO-8859-1",
    names=FIELDS,
)
df_test["target"] = df_test["target"].map({0: 0, 2: 1, 4: 2})
df_test.head()


# PREPROCESSING
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(df_train["text"])
X_test = vectorizer.transform(df_test["text"])
y_train = df_train["target"]
y_test = df_test["target"]

# LOGISTIC REGRESSION
with wandb.init(
    project="project", entity="subsystem3", name="logistic-regression"
) as run:
    config = run.config
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

    # LOG METRICS
    wandb.log(
        {
            "accuracy": accuracy_score(y_test, y_pred_lr),
            "precision": precision_score(
                y_test, y_pred_lr, average="weighted", zero_division=1
            ),
            "recall": recall_score(y_test, y_pred_lr, average="weighted"),
            "f1": f1_score(y_test, y_pred_lr, average="weighted"),
            "lr_confusion_matrix": wandb.plot.confusion_matrix(
                probs=None, y_true=y_test, preds=y_pred_lr, class_names=CLASS_NAMES
            ),
        }
    )

    # SAVE MODEL
    joblib.dump(lr, "models/lr_model.pkl")

    # LOG MODEL
    artifact = wandb.Artifact("lr_model", type="model")
    artifact.add_file("models/lr_model.pkl")
    run.log_artifact(artifact)

# MULTINOMIAL NAIVE BAYES
with wandb.init(
    project="project", entity="subsystem3", name="multinomial-naive-bayes"
) as run:
    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    y_pred_nb = nb.predict(X_test)

    # LOG METRICS
    wandb.log(
        {
            "accuracy": accuracy_score(y_test, y_pred_nb),
            "precision": precision_score(
                y_test, y_pred_nb, average="weighted", zero_division=1
            ),
            "recall": recall_score(y_test, y_pred_nb, average="weighted"),
            "f1": f1_score(y_test, y_pred_nb, average="weighted"),
            "nb_confusion_matrix": wandb.plot.confusion_matrix(
                probs=None, y_true=y_test, preds=y_pred_nb, class_names=CLASS_NAMES
            ),
        }
    )

    # SAVE MODEL
    joblib.dump(nb, "models/nb_model.pkl")

    # LOG MODEL
    artifact = wandb.Artifact("nb_model", type="model")
    artifact.add_file("models/nb_model.pkl")
    run.log_artifact(artifact)

# LINEAR SUPPORT VECTOR MACHINE
with wandb.init(
    project="project", entity="subsystem3", name="linear-support-vector-machine"
) as run:
    config = run.config
    svm = LinearSVC(dual=True)
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)

    # LOG METRICS
    wandb.log(
        {
            "accuracy": accuracy_score(y_test, y_pred_svm),
            "precision": precision_score(
                y_test, y_pred_svm, average="weighted", zero_division=1
            ),
            "recall": recall_score(y_test, y_pred_svm, average="weighted"),
            "f1": f1_score(y_test, y_pred_svm, average="weighted"),
            "svm_confusion_matrix": wandb.plot.confusion_matrix(
                probs=None, y_true=y_test, preds=y_pred_svm, class_names=CLASS_NAMES
            ),
        }
    )

    # SAVE MODEL
    joblib.dump(svm, "models/svm_model.pkl")

    # LOG MODEL
    artifact = wandb.Artifact("svm_model", type="model")
    artifact.add_file("models/svm_model.pkl")
    run.log_artifact(artifact)
