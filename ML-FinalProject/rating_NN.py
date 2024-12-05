import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler

def train(title,director,description,cast):
    df = pd.read_csv('imdb-movies-dataset.csv')

    df['combined_features'] = (
        df['Title'] + ' ' + df['Director'] + ' ' + df['Description'] + ' ' + df['Cast']
    )

    df['combined_features'] = df['combined_features'].fillna('')

    vectorizer = TfidfVectorizer(max_features=5000)  # Limit to top 5000 features
    feature_vectors = vectorizer.fit_transform(df['combined_features'])

    scaler = MinMaxScaler()
    df['Rating'] = scaler.fit_transform(df[['Rating']])

    X_train, X_test, y_train, y_test = train_test_split(
        feature_vectors, df['Rating'], test_size=0.2, random_state=42
    )

    X_train_tensor = torch.tensor(X_train.toarray(), dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.toarray(), dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

    X_train_tensor = torch.nan_to_num(X_train_tensor, nan=0.0, posinf=1e6, neginf=-1e6)
    y_train_tensor = torch.nan_to_num(y_train_tensor, nan=0.0, posinf=1e6, neginf=-1e6)

    class MovieRatingPredictor(nn.Module):
        def __init__(self, input_size):
            super(MovieRatingPredictor, self).__init__()
            self.fc = nn.Sequential(
                nn.Linear(input_size, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1) 
            )

        def forward(self, x):
            return self.fc(x)

    # Initialize the model
    input_size = X_train_tensor.shape[1]
    model = MovieRatingPredictor(input_size)

    criterion = nn.MSELoss()  # Mean Squared Error for regression
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    epochs = 20
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        predictions = model(X_train_tensor)
        loss = criterion(predictions, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    X_test_tensor = torch.nan_to_num(X_test_tensor)
    y_test_tensor = torch.nan_to_num(y_test_tensor)


    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test_tensor)
        test_loss = criterion(test_predictions, y_test_tensor)
        print(f"Test Loss: {test_loss.item():.4f}")

    myData = [[title, director, description, cast]]
   
    myData_combined = [
        ' '.join(myData[0])  # Combine the features into a single string
    ]

    # Transform the text into feature vectors using the pre-trained vectorizer
    myData_transformed = vectorizer.transform(myData_combined)

    # Convert to a PyTorch tensor
    myData_tensor = torch.tensor(myData_transformed.toarray(), dtype=torch.float32)

    # Make predictions
    torch.manual_seed(42)
    np.random.seed(42)

    # After predicting with the model, reverse the scaling on the prediction
    model.eval()
    with torch.no_grad():
        predictions = model(myData_tensor)
        predicted_rating = predictions.item()

    # Reverse the normalization of the predicted rating
    predicted_rating = scaler.inverse_transform([[predicted_rating]])[0][0]
    
    # print(f"Predicted Rating: {predicted_rating:.1f}")
    return predicted_rating
