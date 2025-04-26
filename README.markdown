# Phishing Detection Hackathon Project

## Overview
This project develops a machine learning-based phishing detection system to classify emails as **Safe (Legitimate)** or **Phishing**. Built for the Quantam Breach hackathon on April 25, 2025, it processes email text, trains a Support Vector Machine (SVM) model with TF-IDF features, and predicts phishing attempts with high accuracy (~96%). The system supports both batch predictions (via CSV) and interactive predictions (via scripts and a Flask web app), handling multi-line emails and edge cases like invoice scams.

## Features
- **Data Preprocessing**: Cleans email text, preserves URLs, and handles invalid data for robust feature extraction.
- **Model Training**: Uses SVM with TF-IDF features and class weights to achieve high accuracy and handle imbalanced data.
- **Prediction**:
  - **Batch Mode**: Processes CSV files (e.g., `test_emails.csv`) for bulk predictions.
  - **Interactive Mode**: Supports real-time input via command-line (`predict.py`) or a Flask web app (`app.py`).
- **Web App**: Interactive interface at `http://127.0.0.1:5000` for classifying emails, displaying confidence scores and top keywords.
- **UI Design**: Modern, responsive design with a black, white, and purple color scheme, using Inter font and Tailwind CSS for a professional look.
- **Evaluation**: Achieves ~96% accuracy, with high precision and recall, and fixes for misclassification of neutral emails (e.g., short meeting invites).

## Repository Structure
- `preprocess_data.py`: Preprocesses raw email data (`Phishing_Email.csv`) into `processed_data.csv`.
- `train_model.py`: Trains the SVM model, saving `model.pkl` and `vectorizer.pkl`.
- `predict.py`: Predicts email types from CSV (`test_emails.csv`) or interactive input, saving to `predictions.csv`.
- `app.py`: Flask application for the web app, serving the interactive UI.
- `templates/index.html`: Webpage template with a modern black, white, and purple aesthetic.
- `Phishing_Email.csv`: Input dataset with `Email Text` and `Email Type` (managed with Git LFS).
- `processed_data.csv`: Cleaned dataset with `cleaned_text` and `label`.
- `invalid_rows.csv`: Rows with invalid `Email Type` values.
- `test_emails.csv`: 15 multi-line test emails (8 safe, 7 phishing).
- `predictions.csv`: Prediction results from `predict.py`.
- `model.pkl`, `vectorizer.pkl`: Trained SVM model and TF-IDF vectorizer.

## Setup
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/mmnabeel317/Phishing_Guard.git
   cd Phishing-detector
   ```

2. **Set Up Virtual Environment**:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   - Ensures `pandas`, `scikit-learn`, `joblib`, `flask`, and others are installed.

4. **Verify Files**:
   - Ensure `Phishing_Email.csv`, `test_emails.csv`, scripts, and `templates/` are present.

## Usage
### Preprocess Data
```bash
python preprocess_data.py
```
- **Inputs**: `Phishing_Email.csv`
- **Outputs**: `processed_data.csv`, `invalid_rows.csv`

### Train Model
```bash
python train_model.py
```
- **Inputs**: `processed_data.csv`
- **Outputs**: `model.pkl`, `vectorizer.pkl`
- **Displays**: Accuracy, precision, recall, and F1-score (~96% accuracy).

### Predict Emails
#### Batch Mode
```bash
python predict.py --input test_emails.csv --output predictions.csv
```
- **Inputs**: `test_emails.csv`
- **Outputs**: `predictions.csv`

#### Interactive Mode (Command-Line)
```bash
python predict.py
```
- Enter multi-line emails, type `END_EMAIL` to separate, press Enter twice to finish.
- **Example**:
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

#### Web App
```bash
python app.py
```
- Open `http://127.0.0.1:5000` in a browser.
- Paste an email into the textarea and click "Classify Email" to see the classification, confidence, and top keywords.
- **Example Emails**:
  - **Legitimate**:
    ```
    Subject: Weekly Team Sync
    Hi Team,
    Our next sync is Monday at 9 AM.
    Best,
    Sarah
    ```
  - **Phishing**:
    ```
    Subject: Urgent: Account Verification
    Your account needs verification.
    Click here: http://secure-login.com
    ```

## Challenges and Solutions
- **Challenge**: Misclassification of the “Invoice Overdue” phishing email as Safe.
- **Solution**: Switched to SVM, preserved URLs in preprocessing, and added class weights to handle imbalance.
- **Challenge**: Misclassification of short, neutral Legitimate emails (e.g., meeting invites).
- **Solution**: Cleaned `processed_data.csv` to remove noisy Legitimate emails and added probability calibration to the SVM model.
- **Challenge**: Basic web UI lacking visual appeal.
- **Solution**: Implemented a modern, responsive UI with a black, white, and purple aesthetic using Tailwind CSS and Inter font.

## Results
- **Model Performance**: ~96% accuracy, with high precision and recall for phishing detection.
- **Test Results**: Correctly classified 15/15 emails in `test_emails.csv` and interactive inputs, including edge cases.
- **Web App**: Provides accurate, real-time classification with confidence scores (e.g., ~70-90% for Legitimate, ~95-99% for Phishing) and a polished UI.
- **Artifacts**: All scripts, data, models, and the web app are included for reproducibility.

## Acknowledgments
- Built for the Quantam Breach hackathon .
- Uses `scikit-learn` for machine learning, `Flask` for the web app, and `Tailwind CSS` for styling.

## Contact
- **Repository**: [https://github.com/mmnabeel317/Phishing_Guard](https://github.com/mmnabeel317/Phishing_Guard)
