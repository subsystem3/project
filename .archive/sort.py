import datetime
import glob
import itertools
import json
import logging
import multiprocessing
import os
import pickle
import re
import string
import time
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from functools import wraps
from pprint import pprint
from queue import Queue
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import dotenv
import ipywidgets as widgets
import joblib
import matplotlib.pyplot as plt
import nltk
import numpy as np
import openai
import pandas as pd
import preprocessor
import seaborn as sns
import tensorflow_hub as hub
from adjustText import adjust_text
from colorama import Fore, Style
from gensim.models import Doc2Vec, Word2Vec
from gensim.models.doc2vec import TaggedDocument
from IPython.display import Markdown, clear_output, display, display_html
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from tqdm import tqdm
from wordcloud import WordCloud

import wandb

warnings.filterwarnings("ignore")
