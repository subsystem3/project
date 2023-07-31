# TUTORIALS

<https://www.analyticsvidhya.com/blog/2021/12/sentiment-analysis-on-tweets-with-lstm-for-beginners/>

<https://www.kaggle.com/code/stoicstatic/twitter-sentiment-analysis-for-beginners>

## `wandb`

```python
import datetime
import joblib


def model_Evaluate(model):
    with wandb.init(
        project="project",
        config={"model": model},
        name=f"{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}_{model}",
    ) as run:
        # PREDICT VALUES FROM TEST DATASET
        y_pred = model.predict(X_test)

        # PRINT THE CLASSIFICATION REPORT
        print(classification_report(y_test, y_pred))

        # COMPUTE AND PLOT CONFUSION MATRIX
        cf_matrix = confusion_matrix(y_test, y_pred)
        categories = ["Negative", "Positive"]
        group_names = ["True Neg", "False Pos", "False Neg", "True Pos"]
        group_percentages = [
            "{0:.2%}".format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)
        ]
        labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_names, group_percentages)]
        labels = np.asarray(labels).reshape(2, 2)
        sns.heatmap(
            cf_matrix,
            annot=labels,
            cmap="Blues",
            fmt="",
            xticklabels=categories,
            yticklabels=categories,
        )
        plt.xlabel("Predicted values", fontdict={"size": 14}, labelpad=10)
        plt.ylabel("Actual values", fontdict={"size": 14}, labelpad=10)
        plt.title("Confusion Matrix", fontdict={"size": 18}, pad=20)

        # LOG TO W&B
        wandb.log(
            {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(
                    y_test, y_pred, average="weighted", zero_division=1
                ),
                "recall": recall_score(y_test, y_pred, average="weighted"),
                "f1": f1_score(y_test, y_pred, average="weighted"),
                "roc_auc": roc_auc_score(y_test, y_pred),
                "confusion_matrix": wandb.Image(plt),
            }
        )

        # LOG MODEL
        artifact = wandb.Artifact("model", type="model")
        joblib.dump(model, "model.pkl")
        artifact.add_file("model.pkl")
        run.log_artifact(artifact)
        os.remove("model.pkl")
```
