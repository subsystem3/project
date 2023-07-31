# %% [markdown]
# # Brand Sentiment Analysis of Twitter Posts

# %% [markdown]
# ## Setup

import json

# %%
# STANDARD IMPORTS
import os

# %%
# INSTALL REQUIRED PACKAGES FOR PROJECT
PACKAGES = [
    "matplotlib",
    "numpy",
    "openai",
    "pandas",
    "python-dotenv",
    "seaborn",
    "tweet-preprocessor",
    "wandb",
    "nltk",
    "sklearn",
    "transformers",
    "torch",
]


def handle_packages(package: str) -> None:
    """Quietly installs a package if it is not already installed.

    Args:
        package (str): The name of the package to install.

    Raises:
        ImportError: If the package is not installed, install it.
    """
    try:
        __import__(package)
    except ImportError:
        os.system(f"pip install -q {package}")


for package in PACKAGES:
    handle_packages(package=package)

# %%
# STANDARD DATA MANIPULATION/VISUALIZATION LIBRARIES
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# CONFIGURE
pd.set_option("display.max_colwidth", 80)

# %%
# LOAD ENVIRONMENT VARIABLES
import dotenv

ENVIRONMENT = dotenv.dotenv_values()
for key in ENVIRONMENT:
    os.environ[key] = ENVIRONMENT[key]

# %% [markdown]
# ## Exploratory Data Analysis (EDA)

# %% [markdown]
# ### Sentiment140 Dataset
#
# Sentiment140 is a dataset containing 1.6 million tweets with sentiment labels.
# It allows you to discover the sentiment of a brand, product, or topic on Twitter.
# The data is a CSV with emoticons removed.

# %%
# PULL TRAINING DATASET FROM GIT LFS
# !git lfs pull -I "datasets/training.1600000.processed.noemoticon.csv"

df_train = pd.read_csv(
    "datasets/training.1600000.processed.noemoticon.csv",
    encoding="ISO-8859-1",
    names=["target", "ids", "date", "flag", "user", "text"],
    header=None,
)

# The dataset has 6 fields:
#
# 1. `target` — the polarity of the tweet (0 = negative, 2 = neutral, 4 = positive)
# 1. `id` — the id of the tweet (2087)
# 1. `date` — the date of the tweet (Sat May 16 23:58:44 UTC 2009)
# 1. `flag` — the query (lyx). If there is no query, then this value is NO_QUERY.
# 1. `user` — the user that tweeted (robotickilldozr)
# 1. `text` — the text of the tweet (Lyx is cool)

# %% [markdown]
# ### Class Balance
#
# We show that the dataset is balanced by counting the number of positive and negative tweets.
#
# Balance is important because it means that the model will be trained on an equal number of positive and negative tweets.
# If the dataset was imbalanced, then the model would be trained on more of one class than the other.
# This would result in a model that is biased towards the class with more samples.

# %%
# SHOW SENTIMENT DISTRIBUTION IN TRAINING SET
distribution = df_train["target"].value_counts()
plt.figure(figsize=(8, 6))
sns.barplot(x=distribution.index, y=distribution.values)
plt.title("Distribution of Sentiment in Sentiment140 Training Set")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.xticks([0, 1], ["Negative", "Positive"])

for i, v in enumerate(distribution.values):
    # ADD COUNTS ABOVE BARS
    plt.text(i, v, str(v), ha="center")

plt.show()

# %% [markdown]
# ### Word Cloud

# %%
from wordcloud import WordCloud

wordcloud = WordCloud(
    width=1600,
    height=800,
    background_color="white",
    min_font_size=10,
    max_words=1000,
    collocations=False,
)

wordcloud.generate(" ".join(df_train["text"].tolist()))

plt.figure(figsize=(8, 8))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

# %% [markdown]
# ## Preprocessing

# %% [markdown]
# Preprocessing is traditionally an important step for Natural Language Processing (NLP) tasks. It transforms text into a more digestible form so that machine learning algorithms can perform better.
#
# Preprocessing steps include:
#
# * Lower Casing: Each text is converted to lowercase.
# * Replacing URLs: Links starting with "http" or "https" or "www" are replaced by "URL".
# * Replacing Emojis: Replace emojis by using a pre-defined dictionary containing emojis along with their meaning. (eg: ":)" to "EMOJIsmile")
# * Replacing Usernames: Replace @Usernames with word "USER". (eg: "@Kaggle" to "USER")
# * Removing Non-Alphabets: Replacing characters except Digits and Alphabets with a space.
# * Removing Consecutive letters: 3 or more consecutive letters are replaced by 2 letters. (eg: "Heyyyy" to "Heyy")
# * Removing Short Words: Words with length less than 2 are removed.
# * Removing Stopwords: Stopwords are the English words which does not add much meaning to a sentence. They can safely be ignored without sacrificing the meaning of the sentence. (eg: "the", "he", "have")
# * Lemmatizing: Lemmatization is the process of converting a word to its base form. (e.g: "Great" to "Good")

# %%
# SELECT COLUMNS OF INTEREST
df_train = df_train[["text", "target"]]

# REMAP SENTIMENT LABELS: 0 = negative, 1 = positive (instead of 0 = negative, 4 = positive)
df_train["target"] = df_train["target"].replace(4, 1)

# %%
BRANDS = [
    "google",
    "facebook",
    "microsoft",
    "amazon",
    "apple",
    "walmart",
    "nike",
    "target",
    "starbucks",
    "mcdonalds",
    "netflix",
    "disney",
    "zara",
    "adidas",
    "gucci",
    "verizon",
    "sony",
]

# %%
# VISUALIZE BRAND MENTION FREQUENCY
from collections import Counter

counter = Counter(
    word
    for tweet in df_train["text"]
    for word in tweet.lower().split()
    if word in BRANDS
)

brand_freqs_df = pd.DataFrame(
    counter.items(), columns=["Brand", "Frequency"]
).sort_values("Frequency", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(
    x="Frequency", y="Brand", data=brand_freqs_df, order=brand_freqs_df["Brand"]
)
plt.xticks(rotation=90)
plt.title("Brand Mention Frequency")
plt.show()

# %%
brand_rows = []
for tweet in df_train["text"]:
    # TOKENIZE EACH TWEET, REMOVE HYPHENS, AND CONVERT ALL WORDS TO LOWERCASE
    tweet_tokens = {word.replace("-", "") for word in tweet.lower().split()}
    # FIND THE INTERSECTION OF TOKENIZED TWEETS AND BRANDS
    brands_in_tweet = tweet_tokens.intersection(BRANDS)
    if brands_in_tweet:
        for brand in brands_in_tweet:
            # APPEND A DICTIONARY WITH TWEET TEXT, BRAND PRESENCE FLAG, AND BRAND NAME TO THE LIST
            brand_rows.append({"tweet": tweet, "brand-presence": 1, "brand": brand})
    else:
        # APPEND A DICTIONARY WITH TWEET TEXT, BRAND PRESENCE FLAG, AND "NOBRAND" TO THE LIST
        brand_rows.append({"tweet": tweet, "brand-presence": 0, "brand": "nobrand"})

# CONVERT THE LIST OF DICTIONARIES INTO A PANDAS DATAFRAME
brands_df = pd.DataFrame(brand_rows)

# PRINT THE FIRST FEW ROWS OF THE DATAFRAME
print(brands_df.head())

# %%
# OUTPUT DATASET TO FILE
print(len(brands_df))
brands_df.to_csv("datasets/brands.csv", index=False)

# %%
# DEFINE PARAMETERS FOR DATA GENERATION

MODEL = "gpt-4"

DATASET_SIZE = 10

USER_PROMPT = f"""
    Create some data in the format:

    "tweet"|||brand-presence|||brand

    where:
        tweet is a Twitter post,
        brand-presence is 0 or 1 indicating presence or absence of a brand in the tweet, and
        brand is a brand named in this list: {str(BRANDS)} or nobrand when brand is 0.

    EXAMPLES:
        "@ashman01 My only complaint about Facebook is they've changed it so much it's confusing"|||1|||facebook
        "@juliet I ate some grapes yesterday and I loved them."|||0|||nobrand

    There should be an equal number of tweets with a brand (1) and tweets with no brand (0).

    Create {DATASET_SIZE} data points.
    """

# %%
import openai

openai.api_key = os.environ["OPENAI_API_KEY"]

# GENERATE DATA USING OPENAI'S GPT-4 MODEL
chat_completion = openai.ChatCompletion.create(
    model=MODEL,
    temperature=0,
    messages=[
        {"role": "user", "content": USER_PROMPT},
    ],
)

raw_chat_completion = chat_completion.choices[0].message.content
print(json.dumps(raw_chat_completion.split("\n"), indent=4))

# %%
# PROCESS RAW CONTENT BY SPLITTING IT INTO tweet, brand-presence, AND brand
processed_content = [
    {"tweet": content[0], "brand-presence": int(content[1]), "brand": content[2]}
    for content in (s.split("|||") for s in raw_chat_completion.split("\n"))
]

# PRETTY PRINT PROCESSED CONTENT
print(json.dumps(processed_content, indent=4))

# ADD TO brands_df DATAFRAME
processed_content_df = pd.DataFrame(processed_content)
brands_df = pd.concat([brands_df, processed_content_df], ignore_index=True)

# %%
print(f"brands_df.shape: {brands_df.shape}")

# %%
print(f"brands_df.brand.value_counts(): {brands_df.brand.value_counts()}")

# %%
print(f"brands_df.tail(6): \n{brands_df.tail(6)}")

# %%
plt.figure(figsize=(8, 8))
sns.countplot(x="brand-presence", data=brands_df)
plt.title("Distribution of Brand/No Brand Tweets in Brand Dataset")
plt.ylabel("Count")
plt.xticks([0, 1], ["No Brand", "Brand"])
plt.show()

# %%
import nltk
from nltk.corpus import stopwords

nltk.download("punkt", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("maxent_ne_chunker", quiet=True)
nltk.download("words", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download("stopwords", quiet=True)

STOP_WORDS = set(stopwords.words("english"))

import string

# %%
import preprocessor as p
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer

# REMOVE URLS AND @MENTIONS
p.set_options(p.OPT.URL, p.OPT.MENTION)

tweet_tokenizer = TweetTokenizer()
stemmer = PorterStemmer()


def preprocess(tweet):
    cleaned = p.clean(tweet)
    tokens = tweet_tokenizer.tokenize(cleaned)
    # REMOVE STOPWORDS AND PUNCTUATION
    filtered = [
        word
        for word in tokens
        if not word in STOP_WORDS and not word in string.punctuation
    ]
    # APPLY STEMMING
    stemmed = [stemmer.stem(word) for word in filtered]
    return stemmed


# SHOW SAMPLE TWEETS
tweets_sample = df_train[df_train["target"] == 1].head()
for tweet in tweets_sample["text"]:
    print(f"Original: {tweet}")
    processed_tweet = preprocess(tweet)
    print(f"Processed: {processed_tweet}\n")

# %%
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

vectorizer = CountVectorizer()
X_train, X_test, y_train, y_test = train_test_split(
    brands_df["tweet"], brands_df["brand"], test_size=0.2, random_state=42
)

# %%
from sklearn.naive_bayes import MultinomialNB

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
clf = MultinomialNB()
clf.fit(X_train_vec, y_train)

# %%
from sklearn.metrics import accuracy_score, confusion_matrix

y_pred = clf.predict(X_test_vec)
print("Accuracy: ", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# %%
# ABOUTNESS PROBLEM
test_tweets = [
    "I love to ride my bike",
    "I hate eating apples",
    "Did you google that?",
    "Hello world",
]

tweets_vectorized = vectorizer.transform(test_tweets)

out = clf.predict(tweets_vectorized)

print(out)

# %%
# FILL MISSING VALUES IN 'brand' WITH 'NoBrand'
brands_df["brand"].fillna("NoBrand", inplace=True)

# INITIALIZING CountVectorizer (AKA BAG OF WORDS)
vectorizer = CountVectorizer()

# SPLIT DATA INTO TRAIN AND TEST SETS, 20% DATA IS USED FOR TESTING
X_train, X_test, y_train, y_test = train_test_split(
    brands_df["tweet"], brands_df["brand"], test_size=0.2, random_state=42
)

# TRANSFORM TWEETS INTO NUMERICAL FEATURE VECTORS USING CountVectorizer
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# %%
# IMPORT NECESSARY MODULES
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# DEFINE A LOGISTIC REGRESSION MODEL
log_reg_model = LogisticRegression()

# FIT THE MODEL TO THE TRAINING DATA
log_reg_model.fit(X_train_vectorized, y_train)

# PREDICT ON THE TEST DATA
y_pred = log_reg_model.predict(X_test_vectorized)

# PRINT CLASSIFICATION REPORT
print(classification_report(y_test, y_pred, zero_division=0))

import torch

# %%
# DEFINE TRANSFORMERS MODEL
import transformers
from transformers import (
    AdamW,
    BertModel,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

PRE_TRAINED_MODEL_NAME = "bert-base-cased"
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

# %%
# TOKENIZE TWEETS
sample_txt = brands_df["tweet"][0]
tokens = tokenizer.tokenize(sample_txt)
token_ids = tokenizer.convert_tokens_to_ids(tokens)

print(f" Sentence: {sample_txt}")
print(f"   Tokens: {tokens}")
print(f"Token IDs: {token_ids}")

# %%
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset


# DEFINE PYTORCH DATASET
class BrandDataset(Dataset):
    def __init__(self, tweets, brands, tokenizer, max_len):
        self.tweets = tweets
        self.brands = brands
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, item):
        tweet = str(self.tweets[item])
        brand = self.brands[item]

        encoding = self.tokenizer.encode_plus(
            tweet,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "tweet_text": tweet,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "brands": torch.tensor(brand, dtype=torch.long),
        }


# %%
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
brands_df["brand_encoded"] = encoder.fit_transform(brands_df["brand"])


# %%
# CREATE DATA LOADERS
def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = BrandDataset(
        tweets=df["tweet"].to_numpy(),
        brands=df["brand_encoded"].to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len,
    )

    return DataLoader(ds, batch_size=batch_size, num_workers=4)


BATCH_SIZE = 16
MAX_LEN = 160

train_data_loader = create_data_loader(brands_df, tokenizer, MAX_LEN, BATCH_SIZE)


# %%
# DEFINE THE MODEL ARCHITECTURE
class BrandClassifier(nn.Module):
    def __init__(self, n_classes):
        super(BrandClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        output = self.drop(pooled_output)
        return self.out(output)


# %%
# CREATE THE MODEL INSTANCE
N_CLASSES = len(brands_df["brand"].unique())
model = BrandClassifier(N_CLASSES)
model = model.to(device)

# %%
# TRAINING THE MODEL
EPOCHS = 10

optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
total_steps = len(train_data_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=total_steps
)

loss_fn = nn.CrossEntropyLoss().to(device)


# %%
# DEFINE THE TRAINING FUNCTION
def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model = model.train()

    losses = []
    correct_predictions = 0

    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        brands = d["brands"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, brands)

        correct_predictions += torch.sum(preds == brands)
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)


# %%
# TRAIN THE MODEL
from collections import defaultdict

history = defaultdict(list)
best_accuracy = 0

for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print("-" * 10)

    train_acc, train_loss = train_epoch(
        model, train_data_loader, loss_fn, optimizer, device, scheduler, len(brands_df)
    )

    print(f"Train loss {train_loss} accuracy {train_acc}")

    history["train_acc"].append(train_acc)
    history["train_loss"].append(train_loss)

    if train_acc > best_accuracy:
        torch.save(model.state_dict(), "best_model_state.bin")
        best_accuracy = train_acc

# %%
# PLOT TRAINING ACCURACY AND LOSS
plt.plot(history["train_acc"], label="train accuracy")
plt.title("Training history")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend()
plt.ylim([0, 1])
plt.show()

# %%
# LOAD BEST MODEL
model.load_state_dict(torch.load("best_model_state.bin"))


# %%
# SAMPLE PREDICTIONS
def get_predictions(model, data_loader):
    model = model.eval()

    tweet_texts = []
    predictions = []
    prediction_probs = []

    with torch.no_grad():
        for d in data_loader:
            texts = d["tweet_text"]
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)

            tweet_texts.extend(texts)
            predictions.extend(preds)
            prediction_probs.extend(outputs)

    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    return tweet_texts, predictions, prediction_probs


# %%
y_tweet_texts, y_pred, y_pred_probs = get_predictions(model, train_data_loader)

# %%
# PRINT SAMPLE PREDICTIONS
for i in range(5):
    print(f"Tweet: {y_tweet_texts[i]}")
    print(f"Brand: {encoder.inverse_transform([y_pred[i]])[0]}")
    print()

# %%
# LSTM
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# PREPARE DATA FOR LSTM
tokenizer = Tokenizer(num_words=5000, split=" ")
tokenizer.fit_on_texts(brands_df["tweet"].values)

X = tokenizer.texts_to_sequences(brands_df["tweet"].values)
X = pad_sequences(X)  # padding our text vector so they all have the same length
y = pd.get_dummies(brands_df["brand"]).values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# DEFINE LSTM MODEL
model = Sequential()
model.add(Embedding(5000, 256, input_length=X.shape[1]))
model.add(Dropout(0.3))
model.add(LSTM(256, return_sequences=True, dropout=0.3, recurrent_dropout=0.2))
model.add(LSTM(256, dropout=0.3, recurrent_dropout=0.2))
model.add(Dense(N_CLASSES, activation="softmax"))

# COMPILE MODEL
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# TRAIN MODEL
earlystop = EarlyStopping(
    monitor="val_loss", min_delta=0, patience=3, verbose=0, mode="auto"
)
model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    callbacks=[earlystop],
)

# EVALUATE MODEL
score, acc = model.evaluate(X_test, y_test, verbose=2, batch_size=BATCH_SIZE)
print(f"Score: {score}")
print(f"Validation Accuracy: {acc}")
