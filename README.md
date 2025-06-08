# TwitterSentimentAnalysis

This project is a sentiment analysis model designed to predict the sentiment of Twitter comments as either positive or negative. It is built using machine learning techniques and deployed using Streamlit for an interactive user experience.

[Live website link](https://godwin-twittersentimentanalysis.streamlit.app/)

## 🚀 Features

- **Sentiment Prediction**: Analyzes Twitter comments to determine if they are positive or negative.
- **Interactive Interface**: Deployed using Streamlit, allowing users to input comments and receive real-time sentiment analysis.
- **Data Preprocessing**: Includes text cleaning, tokenization, and vectorization to prepare data for the model.
- **Machine Learning Model**: Utilizes a trained model to classify sentiments based on input text.

## 🛠️ Technologies Used

- **Python**: The primary programming language used for development.
- **scikit-learn**: For building and training the machine learning model.
- **NLTK**: For natural language processing tasks such as tokenization and stemming.
- **pandas**: For data manipulation and analysis.
- **Streamlit**: For creating an interactive web application to deploy the model.

## 📦 Installation Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/GODWIN-projects/TwitterSentimentAnalysis.git
cd TwitterSentimentAnalysis
```

### 2. Install Dependencies

Ensure you have Python installed, then run:

```bash
pip install -r requirements.txt
```

### 3. Download NLTK Data

Some NLTK functionalities require additional data. Run the following in a Python shell:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## 💻 Usage

### Run the Streamlit App

```bash
streamlit run app.py
```

### Interact with the App

- Open your web browser and go to [http://localhost:8501](http://localhost:8501)
- Enter a Twitter comment in the text box and click **"Analyze"** to see the sentiment prediction.

## 🧠 Model Details

- **Algorithm**: The model uses a Logistic Regression classifier trained on a dataset of labeled Twitter comments.
- **Preprocessing**: Text data is cleaned by removing URLs, mentions, and special characters, followed by tokenization and vectorization using TF-IDF.
- **Training**: The model is trained on a dataset of Twitter comments with labeled sentiments (positive/negative).

## 📊 Dataset

The model was trained on the **Sentiment140** dataset, which contains 1.6 million Twitter comments labeled as positive or negative. More details can be found at [Sentiment140](http://help.sentiment140.com/for-students).

## 📈 Results and Accuracy

The model achieves an accuracy of approximately **80%** on the test set, demonstrating its effectiveness in predicting sentiment from Twitter comments.

