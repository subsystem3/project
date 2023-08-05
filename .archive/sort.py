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
from colorama import Fore, Style
from IPython.display import clear_output, display
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
from tqdm import tqdm
from wordcloud import WordCloud

import wandb

warnings.filterwarnings("ignore")
