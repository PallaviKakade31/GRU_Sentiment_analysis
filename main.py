from flask import Flask, request, render_template
from src.utils import SentimentAnalysis


app = Flask(__name__)
sentiment_analy = SentimentAnalysis()


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/SentimentAnalysis",methods=["post"])
def sentiment_Analysis():
    input_data =request.form["sentiment"]
    return sentiment_analy.sentiment_analysis(input_data)


if(__name__=="__main__"):
    app.run(host="0.0.0.0", port=8000)