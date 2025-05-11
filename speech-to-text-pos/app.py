from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import nltk
from nltk import pos_tag, word_tokenize

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

app = Flask(__name__)

# Enable CORS for the /tag route, allowing requests from http://localhost:5500
CORS(app, resources={r"/tag": {"origins": "*"}})

@app.route("/tag", methods=["POST"])
def pos_tagging():
    data = request.get_json()
    text = data.get("text")
    tokens = word_tokenize(text)
    tags = pos_tag(tokens)
    return jsonify({"tags": tags})

if __name__ == "__main__":
    app.run(debug=True)
