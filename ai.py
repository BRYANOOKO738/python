import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Function to respond to greetings
def respond_to_greeting(greeting):
    greetings = ["hello", "hi", "hey", "greetings", "howdy"]
    if greeting.lower() in greetings:
        return "Hello! How can I assist you today?"
    else:
        return "I'm here to help with your student data analysis. Let's get started!"

# Sample student data (Study Hours, Previous Scores -> Final Exam Score)
data = {
    "Study Hours": [2, 4, 6, 8, 10, 12],
    "Previous Scores": [50, 60, 70, 75, 85, 90],
    "Final Exam Score": [55, 65, 78, 80, 88, 95]
}

df = pd.DataFrame(data)

# Split data (Features: Study Hours & Previous Scores, Target: Final Exam Score)
X = df[["Study Hours", "Previous Scores"]]
y = df["Final Exam Score"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple AI model
model = LinearRegression()

# Print training progress
print("Training the model...")
model.fit(X_train, y_train)
print("Model training completed!")

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Predict score for a student who studies 7 hours and scored 55 before
new_student = [[7, 55]]
predicted_score = model.predict(new_student)
print(f"\nPredicted Final Exam Score for a student who studies 7 hours and scored 55 before: {predicted_score[0]:.2f}")

# Visualize the results
plt.figure(figsize=(10, 6))

# Plot actual vs predicted values
plt.scatter(y_test, y_pred, color='blue', label='Actual vs Predicted')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Ideal Prediction')
plt.xlabel("Actual Final Exam Score")
plt.ylabel("Predicted Final Exam Score")
plt.title("Actual vs Predicted Final Exam Scores")
plt.legend()
plt.show()

# Print model coefficients
print("\nModel Coefficients:")
print(f"Intercept: {model.intercept_:.2f}")
print(f"Coefficients: {model.coef_}")

# Interactive greeting response
while True:
    user_input = input("\nSay something (or type 'exit' to quit): ")
    if user_input.lower() == "exit":
        print("Goodbye! Have a great day!")
        break
    else:
        response = respond_to_greeting(user_input)
        print(response)