from flask import Flask, render_template, request
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string


app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
tfid = pickle.load(open("vectorizer.pkl", "rb"))


ps = PorterStemmer()
stop_words = set(stopwords.words("english"))


def transform_text(text):
    text = text.lower()

    text = nltk.word_tokenize(text)

    y = []

    for word in text:
        if word.isalnum():
            y.append(word)

    result = []

    for word in y:
        if word not in stop_words and word not in string.punctuation:
            result.append(ps.stem(word))

    return " ".join(result)



@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    message = request.form["message"]

    transformed_message = transform_text(message)

    vector = tfid.transform([transformed_message]).toarray()

    prediction = model.predict(vector)


    if prediction[0] == 1:
        result = "Spam Email 🚨"
    else:
        result = "Not Spam Email ✅"


    return render_template(
        "index.html",
        prediction=result,
        message=message
    )


if __name__ == "__main__":
    app.run(debug=True)