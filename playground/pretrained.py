import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from transformers import (
    AlbertTokenizer,
    BertTokenizer,
    DistilBertTokenizer,
    ElectraTokenizer,
    GPT2Tokenizer,
    InputExample,
    InputFeatures,
    LongformerTokenizer,
    MobileBertTokenizer,
    RobertaTokenizer,
    TFAlbertForSequenceClassification,
    TFBertForSequenceClassification,
    TFDistilBertForSequenceClassification,
    TFElectraForSequenceClassification,
    TFGPT2ForSequenceClassification,
    TFLongformerForSequenceClassification,
    TFMobileBertForSequenceClassification,
    TFRobertaForSequenceClassification,
    TFXLMForSequenceClassification,
    TFXLNetForSequenceClassification,
    XLMTokenizer,
    XLNetTokenizer,
)
from wandb.keras import WandbCallback

import wandb

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
    (
        TFBertForSequenceClassification,
        BertTokenizer,
        "bert-base-uncased",
    ),
    (
        TFDistilBertForSequenceClassification,
        DistilBertTokenizer,
        "distilbert-base-uncased",
    ),
    (
        TFMobileBertForSequenceClassification,
        MobileBertTokenizer,
        "google/mobilebert-uncased",
    ),
    (
        TFGPT2ForSequenceClassification,
        GPT2Tokenizer,
        "gpt2",
    ),
    (
        TFRobertaForSequenceClassification,
        RobertaTokenizer,
        "roberta-base",
    ),
    (
        TFXLMForSequenceClassification,
        XLMTokenizer,
        "xlm-mlm-enfr-1024",
    ),
    (
        TFXLNetForSequenceClassification,
        XLNetTokenizer,
        "xlnet-large-cased",
    ),
    (
        TFAlbertForSequenceClassification,
        AlbertTokenizer,
        "albert-xxlarge-v2",
    ),
    (
        TFElectraForSequenceClassification,
        ElectraTokenizer,
        "google/electra-large-discriminator",
    ),
    (
        TFLongformerForSequenceClassification,
        LongformerTokenizer,
        "allenai/longformer-large-4096",
    ),
    (
        TFRobertaForSequenceClassification,
        RobertaTokenizer,
        "roberta-large",
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
            padding="max_length",
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

        # Define the optimizer and loss function
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0
        )
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        # Define the metric
        train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

        # Custom training loop
        for epoch in range(2):
            print(f"Start of epoch {epoch+1}")
            for step, (x_batch_train, y_batch_train) in enumerate(train_data):
                with tf.GradientTape() as tape:
                    # Get model outputs
                    model_outputs = model(x_batch_train, training=True)
                    # Extract logits
                    logits = model_outputs.logits
                    loss_value = loss_fn(y_batch_train, logits)
                grads = tape.gradient(loss_value, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))

                # Update training metric
                train_acc_metric.update_state(y_batch_train, logits)

                # Print metrics to the console
                print(
                    f"Step: {step}, Training Accuracy: {train_acc_metric.result().numpy()}, Training Loss: {loss_value.numpy()}"
                )

                # Log metrics to wandb at every step
                wandb.log(
                    {
                        "train_accuracy": train_acc_metric.result().numpy(),
                        "train_loss": loss_value.numpy(),
                    }
                )

            # Reset the metrics at the end of each epoch
            train_acc_metric.reset_states()

        # Evaluate the model
        loss, accuracy = model.evaluate(validation_data)

        # Log the metrics
        wandb.log({"loss": loss, "accuracy": accuracy})

        # Save the model in the SavedModel format
        model.save("model", save_format="tf")
        run.save("model")

        # Log the model
        wandb.save("model.h5")

        # Finish the wandb run
        wandb.finish()
