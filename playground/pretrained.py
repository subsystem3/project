import pandas as pd
import tensorflow as tf
from transformers import (
    BertTokenizer,
    TFBertForSequenceClassification,
    GPT2Tokenizer,
    TFGPT2ForSequenceClassification,
    RobertaTokenizer,
    TFRobertaForSequenceClassification,
    XLMTokenizer,
    TFXLMForSequenceClassification,
    DistilBertTokenizer,
    TFDistilBertForSequenceClassification,
)
from transformers import InputExample, InputFeatures
from sklearn.model_selection import train_test_split
import wandb
from wandb.keras import WandbCallback

# Load data
df = pd.read_csv(
    "brands.csv", names=["tweet", "brand", "brand_presence", "sentiment"], header=0
)

# Split data
X = df["tweet"]
y_brand_presence = df["brand_presence"]
y_sentiment = df["sentiment"]

(
    X_train_brand_presence,
    X_test_brand_presence,
    y_brand_presence_train,
    y_brand_presence_test,
) = train_test_split(X, y_brand_presence, test_size=0.2, random_state=42)

(
    X_train_sentiment,
    X_test_sentiment,
    y_sentiment_train,
    y_sentiment_test,
) = train_test_split(X, y_sentiment, test_size=0.2, random_state=42)

# Define the models and their corresponding tokenizers
models = [
    (TFBertForSequenceClassification, BertTokenizer, "bert-base-uncased"),
    (TFGPT2ForSequenceClassification, GPT2Tokenizer, "gpt2"),
    (TFRobertaForSequenceClassification, RobertaTokenizer, "roberta-base"),
    (TFXLMForSequenceClassification, XLMTokenizer, "xlm-mlm-enfr-1024"),
    (
        TFDistilBertForSequenceClassification,
        DistilBertTokenizer,
        "distilbert-base-uncased",
    ),
]


# Define a function to convert the data to InputExamples
def convert_data_to_examples(data, labels):
    InputExamples = []
    for input_text, label in zip(data, labels):
        InputExamples.append(
            InputExample(guid=None, text_a=input_text, text_b=None, label=label)
        )
    return InputExamples


# Define a function to convert InputExamples to tf.data.Dataset
def convert_examples_to_tf_dataset(examples, tokenizer, max_length=128):
    features = []

    for e in examples:
        input_dict = tokenizer.encode_plus(
            e.text_a,
            add_special_tokens=True,
            max_length=max_length,
            return_token_type_ids=True,
            return_attention_mask=True,
            pad_to_max_length=True,
            truncation=True,
        )

        input_ids, token_type_ids, attention_mask = (
            input_dict["input_ids"],
            input_dict["token_type_ids"],
            input_dict["attention_mask"],
        )

        features.append(
            InputFeatures(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                label=e.label,
            )
        )

    def gen():
        for f in features:
            yield (
                {
                    "input_ids": f.input_ids,
                    "attention_mask": f.attention_mask,
                    "token_type_ids": f.token_type_ids,
                },
                f.label,
            )

    return tf.data.Dataset.from_generator(
        gen,
        (
            {
                "input_ids": tf.int32,
                "attention_mask": tf.int32,
                "token_type_ids": tf.int32,
            },
            tf.int64,
        ),
        (
            {
                "input_ids": tf.TensorShape([None]),
                "attention_mask": tf.TensorShape([None]),
                "token_type_ids": tf.TensorShape([None]),
            },
            tf.TensorShape([]),
        ),
    )


# Loop over the models
for model_class, tokenizer_class, pretrained_weights in models:
    for label, X_train, X_test, y_train, y_test in [
        (
            "brand_presence",
            X_train_brand_presence,
            X_test_brand_presence,
            y_brand_presence_train,
            y_brand_presence_test,
        ),
        (
            "sentiment",
            X_train_sentiment,
            X_test_sentiment,
            y_sentiment_train,
            y_sentiment_test,
        ),
    ]:
        # Initialize wandb
        run = wandb.init(
            project="brand-sentiment", name=f"{pretrained_weights}_{label}"
        )

        # Log the configurations
        config = wandb.config
        config.learning_rate = 3e-5
        config.epochs = 2
        config.batch_size = 32
        config.model = pretrained_weights
        config.task = label

        # Load pretrained model/tokenizer
        model = model_class.from_pretrained(pretrained_weights)
        tokenizer = tokenizer_class.from_pretrained(pretrained_weights)

        # Convert the data to InputExamples
        train_InputExamples = convert_data_to_examples(X_train, y_train)
        test_InputExamples = convert_data_to_examples(X_test, y_test)

        # Convert the InputExamples to tf.data.Dataset
        train_data = convert_examples_to_tf_dataset(
            list(train_InputExamples), tokenizer
        )
        train_data = train_data.shuffle(100).batch(32).repeat(2)

        validation_data = convert_examples_to_tf_dataset(
            list(test_InputExamples), tokenizer
        )
        validation_data = validation_data.batch(32)

        # Compile the model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0
            ),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy("accuracy")],
        )

        # Train the model
        model.fit(
            train_data,
            epochs=2,
            validation_data=validation_data,
            callbacks=[WandbCallback()],
        )

        # Evaluate the model
        loss, accuracy = model.evaluate(validation_data)

        # Log the metrics
        wandb.log({"loss": loss, "accuracy": accuracy})

        # Save the model weights
        model.save_weights("./model_weights.h5")
        run.save("./model_weights.h5")

        # Finish the wandb run
        wandb.finish()
