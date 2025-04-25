import pandas as pd
import re
import os
import joblib
import argparse

def clean_text(text):
    """
    Clean email text by converting to lowercase, removing special characters,
    and normalizing whitespace.
    """
    if not isinstance(text, str) or pd.isna(text):
        return ""  # Handle missing or non-string values
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s:/.-]', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    return text.strip()

def load_artifacts(model_path='model.pkl', vectorizer_path='vectorizer.pkl'):
    """
    Load the trained model and vectorizer.
    Args:
        model_path (str): Path to trained model.
        vectorizer_path (str): Path to TF-IDF vectorizer.
    Returns:
        tuple: Model and vectorizer.
    """
    try:
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        return model, vectorizer
    except Exception as e:
        print(f"Error loading artifacts: {str(e)}")
        raise

def predict_emails(texts, model, vectorizer):
    """
    Predict whether emails are Safe or Phishing.
    Args:
        texts (list or Series): List of email texts.
        model: Trained model.
        vectorizer: TF-IDF vectorizer.
    Returns:
        list: Predicted labels as 'Safe Email' or 'Phishing Email'.
    """
    # Clean texts
    cleaned_texts = [clean_text(text) for text in texts]
    # Filter out empty texts
    valid_texts = [text for text in cleaned_texts if text]
    if not valid_texts:
        return ['Invalid or empty text'] * len(texts)

    # Transform texts to TF-IDF features
    X_tfidf = vectorizer.transform(cleaned_texts)

    # Predict
    predictions = model.predict(X_tfidf)

    # Map numeric predictions to labels
    label_map = {0: 'Safe Email', 1: 'Phishing Email'}
    return [label_map.get(pred, 'Invalid') for pred in predictions]

def process_input(input_file=None):
    """
    Process input from CSV or interactive input.
    Args:
        input_file (str, optional): Path to CSV with 'Email Text' column.
    Returns:
        list: List of email texts.
    """
    if input_file and os.path.exists(input_file):
        try:
            df = pd.read_csv(input_file)
            if 'Email Text' not in df.columns:
                raise ValueError("CSV must contain 'Email Text' column.")
            return df['Email Text'].tolist()
        except Exception as e:
            print(f"Error reading input file: {str(e)}")
            raise
    else:
        print("Enter email text (type 'END_EMAIL' on a new line to separate emails, press Enter twice to finish):")
        texts = []
        current_email = []
        while True:
            line = input()
            if line == "":  # Double Enter to finish
                if current_email:  # Save last email if exists
                    texts.append("\n".join(current_email))
                break
            if line.strip().upper() == "END_EMAIL":  # Delimiter for new email
                if current_email:  # Save current email
                    texts.append("\n".join(current_email))
                    current_email = []
                continue
            current_email.append(line)
        return texts

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Predict phishing emails.")
    parser.add_argument('--input', type=str, help="Path to input CSV with 'Email Text' column")
    parser.add_argument('--output', type=str, default="predictions.csv", 
                       help="Path to save predictions CSV")
    args = parser.parse_args()

    # Load model and vectorizer
    try:
        model, vectorizer = load_artifacts()
    except Exception as e:
        print(f"Failed to load model/vectorizer: {str(e)}")
        exit(1)

    # Get input texts
    try:
        texts = process_input(args.input)
        if not texts:
            print("No input texts provided.")
            exit(1)
    except Exception as e:
        print(f"Failed to process input: {str(e)}")
        exit(1)

    # Predict
    predictions = predict_emails(texts, model, vectorizer)

    # Display results
    print("\nPredictions:")
    for text, pred in zip(texts, predictions):
        print(f"Email: {text[:50].replace('\n', ' ')}... -> {pred}")

    # Save to CSV if output path provided
    if args.output:
        try:
            df_out = pd.DataFrame({'Email Text': texts, 'Prediction': predictions})
            df_out.to_csv(args.output, index=False)
            print(f"Predictions saved to {args.output}")
        except Exception as e:
            print(f"Error saving predictions: {str(e)}")

if __name__ == "__main__":
    main()