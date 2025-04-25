# Phishing Detection Hackathon Project

## Overview
This project develops a machine learning-based phishing detection system to classify emails as `Safe Email` or `Phishing Email`. Built for the [Hackathon Name] on April 25, 2025, it processes email text, trains an SVM model, and predicts phishing attempts with high accuracy. The system handles multi-line emails and supports both batch and interactive predictions.

## Features
- **Data Preprocessing**: Cleans email text, preserving URLs and handling invalid data.
- **Model Training**: Uses SVM with TF-IDF features for robust classification.
- **Prediction**: Supports batch (CSV) and interactive input for real-time phishing detection.
- **Evaluation**: Achieves ~96% accuracy, with fixes for edge cases like invoice scams.

## Repository Structure
- `preprocess_data.py`: Preprocesses raw email data (`test_data.csv`) into `processed_data.csv`.
- `train_model.py`: Trains an SVM model, saving `model.pkl` and `vectorizer.pkl`.
- `predict.py`: Predicts email types from CSV (`test_emails.csv`) or interactive input, saving to `predictions.csv`.
- `Phishing_Emails.csv`: Input dataset with `Email Text` and `Email Type`.
- `processed_data.csv`: Cleaned dataset with `cleaned_text` and `label`.
- `invalid_rows.csv`: Rows with invalid `Email Type` values.
- `test_emails.csv`: 15 multi-line test emails (8 safe, 7 phishing).
- `predictions.csv`: Prediction results.
- `model.pkl`, `vectorizer.pkl`: Trained SVM model and TF-IDF vectorizer.

## Setup
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/mmnabeel317/Phishing_Guard
   cd phishing-detector
   ```
2. **Create Virtual Environment**:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   ```
3. **Install Dependencies**:
   ```bash
   pip install pandas scikit-learn joblib
   ```
4. **Verify Files**:
   Ensure `Phising_Emails.csv`, `test_emails.csv`, and scripts are present.

## Usage
1. **Preprocess Data**:
   ```bash
   python preprocess_data.py
   ```
   - Inputs: `Phising_Emails.csv`
   - Outputs: `processed_data.csv`, `invalid_rows.csv`

2. **Train Model**:
   ```bash
   python train_model.py
   ```
   - Inputs: `processed_data.csv`
   - Outputs: `model.pkl`, `vectorizer.pkl`
   - Displays accuracy, precision, recall, and F1-score (~96% accuracy).

3. **Predict Emails**:
   - **Batch Mode** (using `test_emails.csv`):
     ```bash
     python predict.py --input test_emails.csv --output predictions.csv
     ```
     - Outputs predictions to `predictions.csv`.
   - **Interactive Mode**:
     ```bash
     python predict.py
     ```
     - Enter multi-line emails, type `END_EMAIL` to separate, press Enter twice to finish.
     - Example:
       ```
       Subject: Invoice Overdue
       Dear Client,
       Your invoice #1234 is overdue.
       Pay now to avoid penalties: http://payment-portal.com/invoice
       Contact us if you have questions.
       END_EMAIL
       [Enter]
       [Enter]
       ```



## Challenges and Solutions
- **Challenge**: Misclassification of the “Invoice Overdue” phishing email as `Safe Email`.
- **Solution**: Switched to SVM, preserved URLs in preprocessing, and added class weights to handle imbalance, ensuring correct prediction.

## Results
- **Model Performance**: ~96% accuracy, with high precision and recall for phishing detection.
- **Test Results**: Correctly classified 15/15 emails in `test_emails.csv`, including edge cases.
- **Artifacts**: All scripts, data, and models are included for reproducibility.


## Contact
Repository: (https://github.com/mmnabeel317/Phishing_Guard).
