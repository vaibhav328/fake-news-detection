from flask import Flask, render_template, request, jsonify
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from flask_cors import CORS
# from newsapi import NewsApiClient
from sklearn.feature_extraction.text import TfidfVectorizer
import requests


app = Flask(__name__)
# CORS(app)
# CORS(app, resources={r"/predict": {"origins": "http://127.0.0.1:5500"}})
CORS(app, resources={r"/predict": {"origins": "*"}})
# CORS(app, resources={r"/predict": {"origins": "http://127.0.0.1:5500"}})


# Load the trained models
multinomial_nb_model = joblib.load('multinomial_nb_model.joblib')
passive_aggressive_model = joblib.load('passive_aggressive_model.joblib')

# # Load the fitted TF-IDF vectorizer
vectorizer = joblib.load('tfidf_vectorizer.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)  # Get JSON data from the request
        text_input = data['text']  # Assuming 'text' is the key in the JSON data

        # Preprocess the text_input using the fitted vectorizer
        text_vectorized = vectorizer.transform([text_input])

        # Make predictions using the loaded models
        nb_prediction = multinomial_nb_model.predict(text_vectorized)[0]
        pa_prediction = passive_aggressive_model.predict(text_vectorized)[0]

        # Return the predictions as JSON
        return jsonify({'multinomial_nb_prediction': nb_prediction, 'passive_aggressive_prediction': pa_prediction})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)


# # Replace this with your actual API key from the News API
# news_api_key = '7b152d1b1b744a39a63655468ab99b21'
# # newsapi = NewsApiClient(api_key=news_api_key)

# # Replace this with the actual URL of your Flask app
# # flask_url = 'http://127.0.0.1:5500/fake_news_detetion/platform2.html'



# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.json  # assuming JSON input
#     text_input = data['text']  # assuming 'text' is the key in the JSON data

#     # Fetch news articles from various sources
#     news_api_url = f'https://newsapi.org/v2/everything?q={text_input}&language=en&sortBy=relevancy&apiKey={news_api_key}'
#     response = requests.get(news_api_url)
#     articles = response.json().get('articles', [])

#     if not articles:
#         return jsonify({'error': 'No articles found for the given query'})

#     # Concatenate the content of all articles
#     concatenated_content = ' '.join([article['content'] for article in articles if article['content']])

#     # Preprocess the text_input if needed (e.g., vectorization)
#     vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
#     text_vectorized = vectorizer.transform([concatenated_content])

#     # Make predictions using the loaded models
#     nb_prediction = multinomial_nb_model.predict(text_vectorized)[0]
#     pa_prediction = passive_aggressive_model.predict(text_vectorized)[0]

#     # Return the predictions as JSON
#     return jsonify({'multinomial_nb_prediction': nb_prediction, 'passive_aggressive_prediction': pa_prediction})

# if __name__ == '__main__':
#     app.run(debug=True)