from flask import Flask, render_template, request, jsonify
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

app = Flask(__name__,static_url_path='/static')

# Assuming you have a dataset named 'flipkart.csv' with columns 'Product_name' and 'Review'
df = pd.read_csv('flipkart.csv')
sid = SentimentIntensityAnalyzer()

def analyze_sentiment(review):
    sentiment_score = sid.polarity_scores(review)['compound']
    if sentiment_score > 0.05:
        return 'Positive reviews 🙂'
    elif sentiment_score < -0.05:
        return 'Negative reviews😞 '
    else:
        return 'Neutral reviews 😐'

@app.route('/')
def index():
     return render_template('index.html')
   
@app.route('/analyze', methods=['POST'])
def analyze():
    product_name = request.form['productName']
    product_reviews = df[df['Product_name'] == product_name]['Review']
    sentiments = product_reviews.apply(analyze_sentiment)
    sentiment_counts = sentiments.value_counts().to_dict()
    
    return jsonify(sentiment_counts)

if __name__ == '__main__':
    app.run(debug=True)





