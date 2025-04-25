import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
from sklearn.svm import SVC

def load_data(file_path):
    """
    Load preprocessed data from CSV.
    Args:
        file_path (str): Path to processed_data.csv.
    Returns:
        tuple: Features (cleaned_text) and labels.
    """
    try:
        df = pd.read_csv(file_path)
        if not all(col in df.columns for col in ['cleaned_text', 'label']):
            raise ValueError("CSV must contain 'cleaned_text' and 'label' columns.")
        return df['cleaned_text'], df['label']
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

def train_model(X, y, test_size=0.2, random_state=42):
    """
    Train a Logistic Regression model on text data.
    Args:
        X (Series): Text data (cleaned_text).
        y (Series): Labels (0 or 1).
        test_size (float): Proportion of data for testing.
        random_state (int): Seed for reproducibility.
    Returns:
        tuple: Trained model, vectorizer, and evaluation metrics.
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Convert text to TF-IDF features
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train model
    model = SVC(kernel='linear', class_weight='balanced', probability=True, random_state=random_state)
    model.fit(X_train_tfidf, y_train)

    # Evaluate model
    y_pred = model.predict(X_test_tfidf)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='binary'),
        'recall': recall_score(y_test, y_pred, average='binary'),
        'f1': f1_score(y_test, y_pred, average='binary')
    }

    return model, vectorizer, metrics

def save_artifacts(model, vectorizer, model_path='model.pkl', vectorizer_path='vectorizer.pkl'):
    """
    Save the trained model and vectorizer.
    Args:
        model: Trained model.
        vectorizer: TF-IDF vectorizer.
        model_path (str): Path to save model.
        vectorizer_path (str): Path to save vectorizer.
    """
    try:
        joblib.dump(model, model_path)
        joblib.dump(vectorizer, vectorizer_path)
        print(f"Model saved to {model_path}")
        print(f"Vectorizer saved to {vectorizer_path}")
    except Exception as e:
        print(f"Error saving artifacts: {str(e)}")
        raise

if __name__ == "__main__":
    # Define file paths
    input_file = "processed_data.csv"
    model_path = "model.pkl"
    vectorizer_path = "vectorizer.pkl"

    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Input file {input_file} not found.")
        exit(1)

    # Load data
    X, y = load_data(input_file)

    # Train model
    try:
        model, vectorizer, metrics = train_model(X, y)
        print("Model evaluation metrics:")
        for metric, value in metrics.items():
            print(f"{metric.capitalize()}: {value:.4f}")

        # Save model and vectorizer
        save_artifacts(model, vectorizer, model_path, vectorizer_path)
    except Exception as e:
        print(f"Error during training: {str(e)}")
        exit(1)