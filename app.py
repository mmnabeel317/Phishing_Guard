from flask import Flask, request, render_template
import joblib
import numpy as np
import logging

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Load model and vectorizer
try:
    model = joblib.load('model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    logging.info("Model and vectorizer loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load model or vectorizer: {str(e)}")
    raise

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            email = request.form['email']
            if not email.strip():
                logging.warning("Empty email input.")
                return render_template('index.html', error="Please enter email content.", result=None)

            logging.info(f"Processing email: {email[:50]}...")
            # Transform email text
            email_tfidf = vectorizer.transform([email])
            # Predict
            label = model.predict(email_tfidf)[0]
            probs = model.predict_proba(email_tfidf)[0]
            confidence = float(max(probs))
            label_text = 'Phishing' if label == 1 else 'Legitimate'

            # Get top TF-IDF features
            feature_names = vectorizer.get_feature_names_out()
            scores = email_tfidf.toarray()[0]
            top_indices = np.argsort(scores)[::-1][:5]
            keywords = [feature_names[i] for i in top_indices if scores[i] > 0]
            keywords = keywords[:5] + ['none'] * (5 - len(keywords))

            result = {
                'label': label_text,
                'confidence': f"{confidence:.2%}",
                'keywords': keywords
            }
            logging.info(f"Prediction: {label_text}, Confidence: {confidence}, Keywords: {keywords}")
            return render_template('index.html', result=result, error=None)
        except Exception as e:
            logging.error(f"Error processing email: {str(e)}")
            return render_template('index.html', error=str(e), result=None)
    
    return render_template('index.html', result=None, error=None)

if __name__ == '__main__':
    try:
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        logging.error(f"Failed to start Flask server: {str(e)}")
        raise