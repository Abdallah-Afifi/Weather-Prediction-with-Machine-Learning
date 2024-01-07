import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import joblib

# Load the dataset (we need a dataset with weather features and labels)
# For demonstration, we'll use a dummy dataset
data = {
    'Temperature': [20, 25, 30, 15, 18, 22, 28, 32, 25],
    'Humidity': [60, 70, 75, 50, 55, 65, 80, 85, 70],
    'Wind_Speed': [10, 15, 5, 12, 8, 20, 18, 25, 15],
    'Weather': ['Rainy', 'Sunny', 'Sunny', 'Rainy', 'Rainy', 'Sunny', 'Sunny', 'Rainy', 'Sunny']
}

df = pd.DataFrame(data)

# Label encoding for the 'Weather' column
label_encoder = LabelEncoder()
df['Weather'] = label_encoder.fit_transform(df['Weather'])

# Features (X) and target (y)
X = df.drop('Weather', axis=1)
y = df['Weather']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest Classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test)

# Evaluate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Save the trained model
joblib.dump(classifier, 'weather_prediction_model.joblib')

# Now we can use the saved model for predicting the weather in future data
loaded_model = joblib.load('weather_prediction_model.joblib')
new_data = pd.DataFrame({'Temperature': [27], 'Humidity': [70], 'Wind_Speed': [15]})
prediction = loaded_model.predict(new_data)
print(label_encoder.inverse_transform(prediction))
