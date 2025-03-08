# evaluation.py - Script for evaluating the fine-tuned model.

from sklearn.metrics import accuracy_score, f1_score
from inference import generate_text
from datasets import load_dataset

dataset = load_dataset("imdb")["test"]

def evaluate_model(num_samples=50):
    predictions, labels = [], []
    for i, example in enumerate(dataset.select(range(num_samples))):
        predicted_label = generate_text(example["text"]).lower()
        true_label = "positive" if example["label"] == 1 else "negative"
        predictions.append(predicted_label)
        labels.append(true_label)

    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    print(f"Accuracy: {acc * 100:.2f}%")
    print(f"F1-Score: {f1 * 100:.2f}%")

evaluate_model()
