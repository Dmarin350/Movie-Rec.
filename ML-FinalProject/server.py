from flask import Flask, request, jsonify, send_from_directory
from rating_NN import movie_rater  # Import your model function

app = Flask(__name__)

# Serve the HTML file
@app.route("/")
def home():
    return send_from_directory(".", "index.html")  # Serve the HTML file from the current directory

# Handle predictions
@app.route("/predict", methods=["POST"])
def predict():
    # Extract data from the request
    movie_data = request.json

    try:
        # Prepare the input data
        title = movie_data.get("title", "")
        director = movie_data.get("director", "")
        cast = ", ".join(movie_data.get("cast", []))  # Join cast list
        description = movie_data.get("description", "")
        duration = int(movie_data.get("duration", 0))

        # Call the `movie_rater` function
        features = [title, director, cast, description, duration]
        predicted_rating = movie_rater(features)

        # Return the prediction
        return jsonify({"rating": round(predicted_rating, 2)})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
