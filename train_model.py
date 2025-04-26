import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
from sklearn.calibration import CalibratedClassifierCV

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
        df = df.dropna(subset=['cleaned_text', 'label'])
        df = df[df['cleaned_text'].str.strip() != '']
        return df['cleaned_text'], df['label']
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

def train_model(X, y, test_size=0.2, random_state=42):
    """
    Train an SVM model on text data with calibration.
    Args:
        X (Series): Text data (cleaned_text).
        y (Series): Labels (0 or 1).
        test_size (float): Proportion of data for testing.
        random_state (int): Seed for reproducibility.
    Returns:
        tuple: Trained model, vectorizer, and evaluation metrics.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    base_model = SVC(kernel='linear', class_weight='balanced', C=0.5, random_state=random_state, probability=False)
    model = CalibratedClassifierCV(base_model, method='sigmoid', cv=5)
    model.fit(X_train_tfidf, y_train)

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
    input_file = "processed_data.csv"
    model_path = "model.pkl"
    vectorizer_path = "vectorizer.pkl"

    if not os.path.exists(input_file):
        print(f"Input file {input_file} not found.")
        exit(1)

    X, y = load_data(input_file)

    try:
        model, vectorizer, metrics = train_model(X, y)
        print("Model evaluation metrics:")
        for metric, value in metrics.items():
            print(f"{metric.capitalize()}: {value:.4f}")

        save_artifacts(model, vectorizer, model_path, vectorizer_path)
    except Exception as e:
        print(f"Error during training: {str(e)}")
        exit(1)