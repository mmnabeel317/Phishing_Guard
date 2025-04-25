import pandas as pd
import re
import os
import csv

def clean_text(text):
    """
    Clean email text by converting to lowercase, removing special characters except URLs,
    and normalizing whitespace.
    """
    if not isinstance(text, str) or pd.isna(text):
        return ""  # Handle missing or non-string values
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s:/.-]', ' ', text)  # Keep URL characters
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    return text.strip()

def preprocess_data(input_file, output_file, invalid_file='invalid_rows.csv'):
    """
    Preprocess the email dataset from CSV and save as CSV.
    Args:
        input_file (str): Path to input CSV file (e.g., 'Phishing_Email.csv').
        output_file (str): Path to output CSV file (e.g., 'processed_data.csv').
        invalid_file (str): Path to save rows with invalid Email Type values.
    """
    try:
        df = pd.read_csv(input_file, usecols=['Email Text', 'Email Type'], 
                        low_memory=False, quoting=csv.QUOTE_ALL)
        required_columns = ['Email Text', 'Email Type']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Input file must contain columns: {required_columns}")
        df = df.dropna(subset=['Email Text'])
        df = df[df['Email Text'].str.strip() != '']
        df['cleaned_text'] = df['Email Text'].apply(clean_text)
        df = df[df['cleaned_text'] != '']
        df['Email Type'] = df['Email Type'].astype(str).str.strip().str.title()
        valid_types = ['Safe Email', 'Phishing Email']
        valid_mask = df['Email Type'].isin(valid_types)
        invalid_rows = df[~valid_mask]
        if not invalid_rows.empty:
            invalid_rows.to_csv(invalid_file, index=False)
            print(f"Invalid Email Type rows saved to {invalid_file}")
            print(f"Invalid Email Type values: {invalid_rows['Email Type'].unique()}")
        df = df[valid_mask]
        df['label'] = df['Email Type'].map({'Safe Email': 0, 'Phishing Email': 1})
        if df.empty:
            raise ValueError("No valid rows remain after filtering invalid Email Type values.")
        df_processed = df[['cleaned_text', 'label']]
        df_processed.to_csv(output_file, index=False)
        print(f"Preprocessed data saved to {output_file}")
    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")
        raise

if __name__ == "__main__":
    input_file = "Phishing_Email.csv"
    output_file = "processed_data1.csv"
    invalid_file = "invalid_rows.csv"
    if not os.path.exists(input_file):
        print(f"Input file {input_file} not found.")
        exit(1)
    preprocess_data(input_file, output_file, invalid_file)