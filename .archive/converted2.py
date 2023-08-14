import datetime
import itertools
import json
import os
import pickle
import re
import string
import time
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from pprint import pprint
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import dotenv
import joblib
import matplotlib.pyplot as plt
import nltk
import numpy as np
import openai
import pandas as pd
import preprocessor
import seaborn as sns
from adjustText import adjust_text
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from wordcloud import WordCloud

import wandb

warnings.filterwarnings("ignore")

os.makedirs("report", exist_ok=True)


class Metrics:
    ...


def measure_time(run_name: str):
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            total_time = time.time() - start_time
            metrics.add_metric(f"{run_name}", total_time)
            print(f"{run_name}...{total_time:.2f} seconds")
            return result

        return wrapper

    return decorator


metrics = Metrics()
PACKAGES = [
    "matplotlib",
    "numpy",
    "openai",
    "pandas",
    "python-dotenv",
    "seaborn",
    "tweet-preprocessor",
    "wandb",
]


@measure_time("installing required packages")
def handle_packages(packages: list) -> None:
    for package in packages:
        try:
            __import__(package)
        except ImportError:
            os.system(f"pip install -q {package}")


handle_packages(PACKAGES)
pd.set_option("display.max_colwidth", 80)
plt.style.use("fivethirtyeight")
ENVIRONMENT = dotenv.dotenv_values()
for key in ENVIRONMENT:
    os.environ[key] = ENVIRONMENT[key]
tweets_df = pd.read_csv(
    "datasets/training.1600000.processed.noemoticon.csv",
    encoding="ISO-8859-1",
    names=["sentiment", "ids", "date", "flag", "user", "tweet"],
    header=None,
)
tweets_df = tweets_df[["tweet", "sentiment"]]
tweets_df.sentiment = tweets_df.sentiment.replace(4, 1)
distribution = tweets_df.sentiment.value_counts()
plt.figure(figsize=(8, 6))
sns.barplot(x=distribution.index, y=distribution.values)
plt.title("Distribution of Sentiment in Sentiment140 Training Set")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.xticks([0, 1], ["Negative", "Positive"])
for i, v in enumerate(distribution.values):
    plt.text(i, v, str(v), ha="center")
plt.tight_layout()
plt.savefig("report/sentiment_distribution.png", dpi=300)

plt.show()


@measure_time("generating word cloud from training data")
def generate_wordcloud(df_train):
    wordcloud = WordCloud(
        width=1600,
        height=800,
        background_color="white",
        min_font_size=10,
        max_words=1000,
        collocations=False,
        random_state=42,  # set for idempotency
    )
    wordcloud.generate(" ".join(df_train.tweet.tolist()))
    plt.figure(figsize=(8, 8))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("report/wordcloud.png", dpi=300)
    plt.show()


generate_wordcloud(tweets_df)
technology = [
    "oracle",
    "ibm",
    "lenovo",
    "sony",
    "adobe",
    "apple",
    "facebook",
    "dell",
    "microsoft",
    "google",
    "amazon",
    "nokia",
    "cisco",
    "samsung",
    "intel",
    "asus",
    "tesla",
    "hp",
    "lg",
    "netflix",
]

clothing = [
    "gap",
    "adidas",
    "vans",
    "zara",
    "puma",
    "nike",
    "prada",
    "gucci",
    "chanel",
]

food = [
    "kfc",
    "subway",
    "chipotle",
    "mcdonalds",
    "dominos",
    "chick-fil-a",
    "starbucks",
    "starburst",
    "wendys",
]

entertainment = [
    "hulu",
    "universal",
    "dc",
    "showtime",
    "amc",
    "disney",
    "cbs",
    "marvel",
    "starz",
    "paramount",
    "mgm",
    "pixar",
    "hbo",
    "netflix",
]

drinks = [
    "smirnoff",
    "bacardi",
    "nescafe",
    "corona",
    "tropicana",
    "baileys",
    "sprite",
    "7up",
    "fanta",
    "heineken",
    "guinness",
    "gatorade",
    "starbucks",
    "fiji",
    "coca-cola",
    "pepsi",
]
# BRANDS = technology + clothing + food + entertainment + drinks
BRANDS = [
    "facebook",
    "google",
    "apple",
    "starbucks",
    "disney",
    "microsoft",
    "target",
    "amazon",
    "walmart",
    "sony",
]
counter = Counter(
    word
    for tweet in tweets_df.tweet
    for word in tweet.lower().split()
    if word in BRANDS
)
brand_freqs_df = pd.DataFrame(
    counter.items(), columns=["brand", "frequency"]
).sort_values("frequency", ascending=False)
num_brands = len(brand_freqs_df)
plt.figure(figsize=(10, num_brands * 0.5))
barplot = sns.barplot(
    x="frequency", y="brand", data=brand_freqs_df, order=brand_freqs_df.brand
)
plt.xticks(rotation=90)
plt.title("Brand Mention Frequency")
for i, v in enumerate(brand_freqs_df.frequency):
    barplot.text(v + 3, i + 0.25, str(v))
plt.tight_layout()
barplot.figure.savefig("report/brand_mention_frequency.png", dpi=300)

plt.show()

category_names = ["technology", "clothing", "food", "entertainment", "drinks"]
categories = [
    (technology, "technology"),
    (clothing, "clothing"),
    (food, "food"),
    (entertainment, "entertainment"),
    (drinks, "drinks"),
]

category_counts = dict.fromkeys(category_names, 0)

for brand, count in counter.items():
    for category, category_name in categories:
        if brand in category:
            category_counts[category_name] += count
category_df = pd.DataFrame.from_dict(
    category_counts, orient="index", columns=["frequency"]
).reset_index()
category_df.columns = ["category", "frequency"]
category_df = category_df.sort_values("frequency", ascending=False).reset_index()
plt.figure(figsize=(8, 6))
ax = sns.barplot(x="category", y="frequency", data=category_df)
ax.set_title("Brand Mentions by Category")
ax.set_ylabel("Frequency")
for i, row in category_df.iterrows():
    ax.text(i, row.frequency + 0.5, row.frequency, ha="center")
plt.tight_layout()
plt.savefig("report/brand_category_mentions.png", dpi=300)

plt.show()


brand_df = pd.DataFrame.from_dict(counter, orient="index", columns=["frequency"])
brand_df.frequency = brand_df.frequency.astype(int)


@measure_time("plotting brands within category")
def plot_brand_data(
    df: pd.DataFrame,
    filter_condition: Union[List[bool], pd.Series],
    label: str,
) -> None:
    """Plots a horizontal bar chart of the counts in a dataframe, filtered by condition.

    Args:
        df (pd.DataFrame): The dataframe to plot data from.
        filter_condition (Union[List[bool], pd.Series]): A boolean list or series used to filter the dataframe.
        label (str): The label to use for the y-axis.
    """
    filtered_df = df.loc[filter_condition]
    sorted_df = filtered_df.sort_values("frequency", ascending=False)
    plt.figure(figsize=(10, len(sorted_df) * 0.5))
    sns.barplot(x="frequency", y=sorted_df.index, data=sorted_df, orient="h")
    plt.ylabel("Brand")
    plt.title(f"Brand Mentions by Category ({label})")
    for i, v in enumerate(sorted_df.frequency):
        plt.text(v + 1, i, str(v))
    plt.tight_layout()
    plt.savefig(f"report/brand_data_{label}.png", dpi=300)
    plt.show()


# CALL FUNCTION FOR EACH CATEGORY
# plot_brand_data(brand_df, technology, "Technology")
# plot_brand_data(brand_df, clothing, "Clothing")
# plot_brand_data(brand_df, food, "Food")
# plot_brand_data(brand_df, entertainment, "Entertainment")
# plot_brand_data(brand_df, drinks, "Drinks")


# ### Process Tweets to Identify Brand Mentions


@measure_time("creating brands dataframe")
def create_brand_dataframe(tweets_df: pd.DataFrame, brands: List[str]) -> pd.DataFrame:
    """
    Create a new dataframe with brand labels.

    Args:
        tweets_df (pd.DataFrame): The dataframe containing tweets and sentiments.
        brands (List[str]): The list of brands to be searched in the tweets.

    Returns:
        pd.DataFrame: The new dataframe with brand labels.
    """
    brand_rows = []
    non_brand_counter = 0
    brand_counter = 0
    for tweet, sentiment in zip(tweets_df.tweet, tweets_df.sentiment):
        tweet_tokens = tweet.lower().split()
        brand_found = False

        for word in tweet_tokens:
            word = word.replace("-", "")
            for brand in BRANDS:
                if word == brand:
                    brand_rows.append(
                        {"tweet": tweet, "brand": brand, "sentiment": sentiment}
                    )
                    brand_found = True
                    brand_counter += 1
                    break

            if brand_found:
                break

        if not brand_found and non_brand_counter < brand_counter:
            brand_rows.append(
                {"tweet": tweet, "brand": "nobrand", "sentiment": sentiment}
            )
            non_brand_counter += 1

    brands_df = pd.DataFrame(brand_rows, columns=["tweet", "brand", "sentiment"])
    return brands_df


brands_df = create_brand_dataframe(tweets_df, BRANDS)
print(brands_df.head())
print(len(brands_df))
brands_df.to_csv("datasets/brands.csv", index=False)
MODEL = "gpt-4"
DATASET_SIZE = 50
USER_PROMPT = f"""
    Create some data in the format:

    "tweet"|||brand|||sentiment

    where:
        tweet is a Twitter post,
        brand is a brand named in this list: {str(BRANDS)} or 'nobrand' when no brand is present, and
        sentiment is 0 or 1 indicating negative (0) or positive (1) sentiment of the tweet.

    EXAMPLES:
        "@ashman01 My only complaint about Facebook is they've changed it so much it's confusing"|||facebook|||0
        "@juliet_user I ate some grapes yesterday and I loved them."|||nobrand|||1

    There should be an equal number of tweets with a brand and tweets with no brand, and a balanced distribution
    of sentiment labels.

    Create {DATASET_SIZE} data points.
    """

USER_PROMPT_BRANDS_ONLY = f"""
    Create some data in the format:

    "tweet"|||brand|||sentiment

    where:
        tweet is a Twitter post,
        brand is a brand named in this list: {str(BRANDS)}, and
        sentiment is 0 or 1 indicating negative (0) or positive (1) sentiment of the tweet.

    EXAMPLES:
        "@ashman01 My only complaint about Facebook is they've changed it so much it's confusing"|||facebook|||0
        "@juliet_user I went to the Apple store yesterday and I loved it."|||apple|||1

    There a balanced distributionof sentiment labels.

    Create {DATASET_SIZE} data points.
    """


@measure_time("generating data with GPT-4")
def generate_data(user_prompt: str, model: Optional[str] = "gpt-4") -> str:
    openai.api_key = os.environ["OPENAI_API_KEY"]
    chat_completion = openai.ChatCompletion.create(
        model=model,
        temperature=0.2,
        messages=[
            {"role": "user", "content": user_prompt},
        ],
    )
    raw_chat_completion = chat_completion.choices[0].message.content
    return raw_chat_completion


@measure_time("saving generated data")
def save_generated_data(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)


@measure_time("loading generated data")
def load_generated_data(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


# GENERATE DATA WITH GPT-4 (DELETE FILE TO REGENERATE)
generated_data_file = "cache/generated_data.pkl"
raw_chat_completion = generate_data(USER_PROMPT_BRANDS_ONLY)
# save_generated_data(raw_chat_completion, generated_data_file)
# if os.path.exists(generated_data_file):
#     raw_chat_completion = load_generated_data(generated_data_file)
# else:
#     raw_chat_completion = generate_data(USER_PROMPT)
#     save_generated_data(raw_chat_completion, generated_data_file)

print(json.dumps(raw_chat_completion.split("\n"), indent=4))
print(len(raw_chat_completion))
raw_chat_completion += "\n"
raw_chat_completion += generate_data(USER_PROMPT_BRANDS_ONLY)
print(json.dumps(raw_chat_completion.split("\n"), indent=4))
print(len(raw_chat_completion))
raw_content = raw_chat_completion.split("\n")
processed_content = [s.split("|||") for s in raw_content]
processed_content_df = pd.DataFrame(
    processed_content, columns=["tweet", "brand", "sentiment"]
)

brands_df = pd.concat([brands_df, processed_content_df], ignore_index=True)
print(json.dumps(processed_content, indent=4))
print(f"brands_df.shape: {brands_df.shape}")
print(f"brands_df.brand.value_counts(): {brands_df.brand.value_counts()}")
print(f"brands_df.tail(6): \n{brands_df.tail(6)}")
brand_presence = brands_df["brand"].apply(
    lambda x: "No Brand" if x == "nobrand" else "Brand"
)
print(brand_presence.value_counts())
counts = brand_presence.value_counts().reset_index()
plt.figure(figsize=(8, 8))
ax = sns.barplot(x="index", y=brand_presence.name, data=counts)
plt.title("Presence of Brand in Tweets")
plt.ylabel("Count")
for p in ax.patches:
    ax.text(
        p.get_x() + p.get_width() / 2.0,
        p.get_height(),
        "%d" % int(p.get_height()),
        fontsize=12,
        color="black",
        ha="center",
        va="bottom",
    )
plt.tight_layout()
plt.savefig("report/brand_count.png", dpi=300)
plt.show()
brand_counts = brands_df["brand"].value_counts()
print(brand_counts)
plt.figure(figsize=(15, 12))
plt.pie(
    brand_counts,
    labels=brand_counts.index,
    autopct="%1.1f%%",
    startangle=140,
    pctdistance=0.85,
    labeldistance=1,
)
plt.axis("equal")
plt.title("Proportion of Tweets Referencing Each Brand")
plt.tight_layout()
plt.savefig("report/proportion_brands_in_tweets.png", dpi=300)
plt.show()
nltk.download("punkt", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("maxent_ne_chunker", quiet=True)
nltk.download("words", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download("stopwords", quiet=True)
STOP_WORDS = set(stopwords.words("english"))
preprocessor.set_options(preprocessor.OPT.URL, preprocessor.OPT.MENTION)
STEMMER = SnowballStemmer("english")
STOP_WORDS_SET = set(STOP_WORDS)
PUNCTUATION_SET = set(string.punctuation)


def preprocess(tweet: str) -> list:
    cleaned = preprocessor.clean(tweet)
    tokens = re.findall(r"\b\w+\b", cleaned)
    filtered = [
        word
        for word in tokens
        if word not in STOP_WORDS_SET and word not in PUNCTUATION_SET
    ]
    stemmed = [STEMMER.stem(word) for word in filtered]
    return stemmed


@measure_time("preprocessing training data")
def apply_preprocessing(df):
    with ThreadPoolExecutor() as executor:
        df["processed_text"] = list(executor.map(preprocess, df.tweet))
    return df


@measure_time("saving preprocessed data")
def save_preprocessed(df, filename):
    with open(filename, "wb") as f:
        pickle.dump(df, f)


@measure_time("loading preprocessed data")
def load_preprocessed(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


preprocessed_file = "cache/preprocessed_train.pkl"
if os.path.exists(preprocessed_file):
    tweets_df = load_preprocessed(preprocessed_file)
else:
    tweets_df = apply_preprocessing(tweets_df)
    save_preprocessed(tweets_df, preprocessed_file)

tweets_sample = tweets_df[tweets_df.sentiment == 1].head()
for index, row in tweets_sample.iterrows():
    print(f"ORIGINAL: {row.tweet}")
    print(f"PROCESSED: {row.processed_text}\n")

print("\n".join([str(text) for text in tweets_df.processed_text.head()]))


@measure_time("getting word sentiment frequencies")
def get_word_sentiment_frequencies(df):
    freqs = {}
    for y, tweet in zip(df.sentiment.values.tolist(), df.processed_text):
        for word in tweet:
            pair = (word, y)
            freqs[pair] = freqs.get(pair, 0) + 1

    freqs_sorted = dict(sorted(freqs.items(), key=lambda x: x[1], reverse=True))

    return freqs_sorted


freqs_sorted = get_word_sentiment_frequencies(tweets_df)
print(f"NUMBER OF UNIQUE WORD-SENTIMENT PAIRS: {len(freqs_sorted)}")


@measure_time("printing sample preprocessed text")
def print_around_average(freqs_sorted: Dict[Tuple[Any, int], int]) -> None:
    print("\n".join(f"{k}: {v}" for k, v in itertools.islice(freqs_sorted.items(), 5)))
    avg_freq = sum(freqs_sorted.values()) / len(freqs_sorted)
    below_avg_index = next(
        i for i, v in enumerate(freqs_sorted.values()) if v < avg_freq
    )
    start_index = below_avg_index - 2  # 2 items before the average
    end_index = below_avg_index + 3  # 2 items after the average
    around_avg_items = list(freqs_sorted.items())[start_index:end_index]
    print("\n".join(f"{k}: {v}" for k, v in around_avg_items))
    print("\n".join(f"{k}: {v}" for k, v in list(freqs_sorted.items())[-5:]))


print_around_average(freqs_sorted)


@measure_time("generating sentiment counts plot")
def generate_sentiment_counts_plot(freqs_sorted, n=50):
    word_list = list(set(word for word, sentiment in freqs_sorted.keys()))
    data = [
        [word, freqs_sorted.get((word, 1), 0), freqs_sorted.get((word, 0), 0)]
        for word in word_list
        if word != "i"
    ]

    data = sorted(data, key=lambda x: x[1] + x[2], reverse=True)[:n]

    x = np.log([x[1] + 1 for x in data])
    y = np.log([x[2] + 1 for x in data])

    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=x, y=y)
    plt.title(f"Positive and Negative Counts of {n} Most Frequent Word Stems")
    plt.xlabel("Log Positive count")
    plt.ylabel("Log Negative count")
    texts = []
    for i in range(len(data)):
        texts.append(plt.text(x[i], y[i], data[i][0], fontsize=12))

    adjust_text(texts)

    min_val = min(np.min(x), np.min(y))
    max_val = max(np.max(x), np.max(y))
    plt.plot([min_val, max_val], [min_val, max_val], color="red")
    plt.tight_layout()
    plt.savefig("report/positive_and_negative_counts.png", dpi=300)

    plt.show()


generate_sentiment_counts_plot(freqs_sorted)

wandb.finish()
wandb.login(key=os.getenv("WANDB_API_KEY"))


class BrandClassifier:
    """A generic brand classifier."""

    @measure_time("initializing classifier")
    def __init__(
        self, classifier, vectorizer: CountVectorizer = CountVectorizer()
    ) -> None:
        self.model = Pipeline([("vectorizer", vectorizer), ("classifier", classifier)])
        datetime_stamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        classifier_name = self.model.named_steps["classifier"].__class__.__name__
        wandb.init(
            project="brand-sentiment-analysis",
            name=f"{datetime_stamp}-{classifier_name}",
        )

    @measure_time("training classifier")
    def fit(self, X_train: List[str], y_train: List[str]) -> None:
        self.model.fit(X_train, y_train)

    @measure_time("predicting sentiment")
    def predict(self, X: List[str]) -> List[str]:
        return self.model.predict(X)

    @measure_time("evaluating classifier")
    def evaluate(self, X: List[str], y: List[str]) -> Tuple[float, float, float, float]:
        y_pred = self.predict(X)
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        return accuracy, precision, recall, f1

    @measure_time("printing classifier results")
    def print_scores(
        self, scores: Tuple[float, float, float, float], data_name: str
    ) -> None:
        accuracy, precision, recall, f1 = scores
        print(f"{data_name} Accuracy: {accuracy:.4f}")
        print(f"{data_name} Precision: {precision:.4f}")
        print(f"{data_name} Recall: {recall:.4f}")
        print(f"{data_name} F1: {f1:.4f}")

    @measure_time("finishing run")
    def finish(self, scores: Tuple[float, float, float, float]) -> None:
        classifier_name = self.model.named_steps["classifier"].__class__.__name__
        accuracy, precision, recall, f1 = scores

        artifact = wandb.Artifact(
            classifier_name,
            type="model",
            description="Trained model for brand sentiment analysis",
            metadata={
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            },
        )
        joblib.dump(self.model, "model.pkl")
        artifact.add_file("model.pkl", name="classifier_name")
        wandb.log_artifact(artifact)
        os.remove("model.pkl")
        wandb.finish()

    @measure_time("splitting data")
    def split_data(
        self, X: List[str], y: List[str]
    ) -> Tuple[List[str], List[str], List[str], List[str], List[str], List[str]]:
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )

        return X_train, y_train, X_test, y_test, X_val, y_val

    @measure_time("running classifier")
    def run(self, X: List[str], y: List[str]) -> None:
        X_train, y_train, X_test, y_test, X_val, y_val = self.split_data(X, y)
        self.fit(X_train, y_train)
        val_scores = self.evaluate(X_val, y_val)
        self.print_scores(val_scores, "Validation")
        test_scores = self.evaluate(X_test, y_test)
        self.print_scores(test_scores, "Test")
        self.finish(test_scores)


BrandClassifier(MultinomialNB()).run(brands_df.tweet, brands_df.brand_presence)

# Brand classification using Multinomial Naive Bayes using brand (name) as the target
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(brands_df.tweet)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(brands_df.brand)
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)
model = MultinomialNB()
model.fit(xtrain, ytrain)

predictions = model.predict(xtest)
predicted_brand_names = label_encoder.inverse_transform(predictions)

accuracy = accuracy_score(ytest, predictions)
precision_micro = precision_score(ytest, predictions, average="micro")
precision_macro = precision_score(ytest, predictions, average="macro")
precision_weighted = precision_score(ytest, predictions, average="weighted")
# recall = recall_score(ytest, predictions)
# f1 = f1_score(ytest, predictions)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision (micro average): {precision_micro:.4f}")
print(f"Precision (macro average): {precision_macro:.4f}")
print(f"Precision (weighted average): {precision_weighted:.4f}")
# print(f"Recall: {recall:.4f}")
# print(f"F1: {f1:.4f}")

class_labels = label_encoder.inverse_transform(range(len(label_encoder.classes_)))
print(class_labels)
cm = confusion_matrix(ytest, predictions)
plt.figure(figsize=(14, 12))
sns.heatmap(
    cm, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Quick test
new_tweets = [
    "apple is the best company",
    "I love apples",
    "Apples are the worst",
    "I like apple?",
]

new_tweets_vectorized = vectorizer.transform(new_tweets)
test_prediction = model.predict(new_tweets_vectorized)
new_predicted_brand_name = label_encoder.inverse_transform(test_prediction)
print(new_predicted_brand_name)
BrandClassifier(LinearSVC()).run(brands_df.tweet, brands_df.brand_presence)
BrandClassifier(LogisticRegression()).run(brands_df.tweet, brands_df.brand_presence)
metrics.print_metrics()
