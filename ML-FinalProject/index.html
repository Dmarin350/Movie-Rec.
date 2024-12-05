<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Movie Rating Predictor</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      background-color: #141414;
      color: #ffffff;
      display: flex;
      justify-content: center;
      align-items: center;
      height: 100vh;
      margin: 0;
    }

    .container {
      background: #222222;
      padding: 30px;
      border-radius: 8px;
      box-shadow: 0 5px 15px rgba(0, 0, 0, 0.5);
      width: 100%;
      max-width: 500px;
      text-align: center;
    }

    h1 {
      color: #e50914;
      margin-bottom: 20px;
    }

    .form-group {
      margin-bottom: 20px;
      text-align: left;
    }

    label {
      font-weight: bold;
      display: block;
      margin-bottom: 8px;
    }

    input, textarea, button {
      width: 100%;
      padding: 10px;
      border-radius: 4px;
      border: 1px solid #333333;
      background-color: #141414;
      color: #ffffff;
      margin-top: 5px;
    }

    button {
      background-color: #e50914;
      border: none;
      cursor: pointer;
      font-weight: bold;
      transition: all 0.3s ease;
    }

    button:hover {
      background-color: #f40612;
      transform: scale(1.05);
    }

    #output {
      margin-top: 20px;
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>🎥 Movie Rating Predictor</h1>
    <form id="movie-form">
      <div class="form-group">
        <label for="title">🎬 Movie Title</label>
        <input type="text" id="title" placeholder="Enter movie title" required>
      </div>

      <div class="form-group">
        <label for="director">🎥 Director</label>
        <input type="text" id="director" placeholder="Enter director's name" required>
      </div>

      

      <div class="form-group">
        <label for="description">📝 Description</label>
        <textarea id="description" rows="4" placeholder="Enter movie description" required></textarea>
      </div>

      <div class="form-group">
        <label for="cast">🧑‍🎤 Cast Members</label>
        <textarea id="cast" rows="3" placeholder="Enter cast members separated by commas" required></textarea>
      </div>

      <button type="button" id="predict-btn">Predict Rating</button>
    </form>
    <div id="output">
      <h2>⭐ Predicted Rating:</h2>
      <p id="rating-result">-</p>
    </div>
  </div>

  <script>
    document.getElementById("predict-btn").addEventListener("click", async () => {
      const title = document.getElementById("title").value;
      const director = document.getElementById("director").value;
      const cast = document.getElementById("cast").value.split(",").map(member => member.trim());
      const description = document.getElementById("description").value;
     

      const movieData = { title, director, cast, description };

      try {
        const response = await fetch("/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(movieData),
        });

        const result = await response.json();
        document.getElementById("rating-result").textContent = result.rating || "Error predicting rating!";
      } catch (error) {
        console.error("Error:", error);
        document.getElementById("rating-result").textContent = "Error connecting to server!";
      }
    });
  </script>
</body>
</html>

</body>
</html>
