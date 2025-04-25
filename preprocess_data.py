import pandas as pd
import re
import os
import csv

def clean_text(text):
    """
    Clean email text by converting to lowercase, removing special characters,
    and normalizing whitespace.
    """
    if not isinstance(text, str) or pd.isna(text):
        return ""  # Handle missing or non-string values
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s]', ' ', text)  # Remove special characters
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
        # Read CSV with proper quoting to handle commas/quotes in Email Text
        df = pd.read_csv(input_file, usecols=['Email Text', 'Email Type'], 
                        low_memory=False, quoting=csv.QUOTE_ALL)

        # Verify required columns
        required_columns = ['Email Text', 'Email Type']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Input file must contain columns: {required_columns}")

        # Remove rows with missing or empty Email Text
        df = df.dropna(subset=['Email Text'])
        df = df[df['Email Text'].str.strip() != '']

        # Clean email text
        df['cleaned_text'] = df['Email Text'].apply(clean_text)

        # Remove rows where cleaned text is empty
        df = df[df['cleaned_text'] != '']

        # Normalize Email Type: strip whitespace, standardize case
        df['Email Type'] = df['Email Type'].astype(str).str.strip().str.title()

        # Identify valid and invalid rows
        valid_types = ['Safe Email', 'Phishing Email']
        valid_mask = df['Email Type'].isin(valid_types)
        invalid_rows = df[~valid_mask]

        # Save invalid rows for debugging
        if not invalid_rows.empty:
            invalid_rows.to_csv(invalid_file, index=False)
            print(f"Invalid Email Type rows saved to {invalid_file}")
            print(f"Invalid Email Type values: {invalid_rows['Email Type'].unique()}")

        # Keep only valid rows
        df = df[valid_mask]

        # Encode labels: Safe Email -> 0, Phishing Email -> 1
        df['label'] = df['Email Type'].map({'Safe Email': 0, 'Phishing Email': 1})

        # Check if any rows remain
        if df.empty:
            raise ValueError("No valid rows remain after filtering invalid Email Type values.")

        # Select relevant columns
        df_processed = df[['cleaned_text', 'label']]

        # Save to CSV
        df_processed.to_csv(output_file, index=False)
        print(f"Preprocessed data saved to {output_file}")

    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")
        raise

if __name__ == "__main__":
    # Define file paths
    input_file = "Phishing_Email.csv"
    output_file = "processed_data.csv"
    invalid_file = "invalid_rows.csv"

    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Input file {input_file} not found.")
        exit(1)

    # Run preprocessing
    preprocess_data(input_file, output_file, invalid_file)