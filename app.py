import numpy as np
from flask import Flask, request, jsonify, render_template
from modules import preprocessing, predict_tags, final_preprocessing

# Create flask app
flask_app = Flask(__name__)

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict/", methods = ["POST"])
def predict():
    title = request.form['title']
    body = request.form['body']
    X= title + body
    doc = preprocessing(X, rejoin=True)
    doc =final_preprocessing(doc)
    prediction = predict_tags(doc)

    # Inverse multilabel binarizer
    #tags_predict = multilabel_binarizer.inverse_transform(prediction)
    return render_template("index.html", prediction_text = "Tag(s) prédit(s) : {}".format(prediction))

if __name__ == "__main__":
    flask_app.run(debug=True)


