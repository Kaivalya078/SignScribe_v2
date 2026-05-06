import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import os

# Paths — always relative to THIS file, not cwd
_PROJECT_ROOT = Path(__file__).resolve().parents[1]

def train_and_evaluate_model(dataset_path, model_output_path, results_output_path):
    # Load the dataset.
    df = pd.read_csv(dataset_path)
    print(f"Loaded {len(df)} samples from {dataset_path}")

    # Separate features (X) and labels (y).
    X = df.drop('label', axis=1)
    y = df['label']

    # Drop classes with too few samples for stratified split.
    counts = y.value_counts()
    valid = counts[counts >= 2].index
    dropped = [c for c in counts.index if c not in valid]
    if dropped:
        print(f"WARNING: Dropping classes with < 2 samples: {dropped}")
        mask = y.isin(valid)
        X, y = X[mask], y[mask]
        print(f"Remaining samples: {len(X)}")

    # Split data into training and testing sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Initialize and train the Random Forest model.
    print("Training the Random Forest model on landmark data...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    print("Model training complete.")

    # Evaluate the model.
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy on Test Set: {accuracy * 100:.2f}%")

    # Save the trained model.
    Path(model_output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(model_output_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to: {model_output_path}")

    # Generate and save the confusion matrix.
    conf_matrix = confusion_matrix(y_test, y_pred, labels=sorted(y.unique()))
    plt.figure(figsize=(14, 12))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=sorted(y.unique()), yticklabels=sorted(y.unique()))
    plt.title('Confusion Matrix (Random Forest Landmark Model)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()

    os.makedirs(results_output_path, exist_ok=True)
    plot_path = os.path.join(results_output_path, 'confusion_matrix_landmark.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Confusion matrix saved to: {plot_path}")


if __name__ == '__main__':
    train_and_evaluate_model(
        dataset_path=str(_PROJECT_ROOT / "dataset" / "asl_landmarks.csv"),
        model_output_path=str(_PROJECT_ROOT / "checkpoints" / "asl_landmark_model.pkl"),
        results_output_path=str(_PROJECT_ROOT / "results"),
    )
