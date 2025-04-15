import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import random

# Create sample training data
greetings = [
    "hello", "hi", "hey", "good morning", "good afternoon", "good evening",
    "howdy", "what's up", "sup", "hello there", "hi there", "hey there",
    "greetings", "yo", "hiya", "morning", "afternoon", "evening",
    "nice to meet you", "pleased to meet you", "hi friend", "hello friend",
    "hi everyone", "hello everyone", "hey everyone", "hi all", "hello all",
    "how are you", "how are you doing", "how's it going", "how do you do",
    "how are things", "what's new", "what's going on"
]

non_greetings = [
    "what time is it", "where are you", "what's your name", "help me",
    "can you help", "i need assistance", "tell me about yourself",
    "what can you do", "what's the weather", "goodbye", "bye", "see you",
    "thanks", "thank you", "appreciate it", "can you explain",
    "tell me a joke", "what's your favorite color", "who created you",
    "where are you from", "i'm bored", "let's talk", "i have a question",
    "what's the answer", "tell me more", "i don't understand", "explain please",
    "when were you made", "who made you", "what is your purpose", "search for",
    "find information about", "where can i find", "i'm looking for"
]

# Create responses for greetings
greeting_responses = [
    "Hello! How can I help you today?",
    "Hi there! What can I do for you?",
    "Hey! Nice to see you!",
    "Greetings! How are you doing?",
    "Hello! I'm here to assist you.",
    "Hi! What brings you here today?",
    "Hey there! How can I be of service?",
    "Good day! What do you need help with?"
]

# Create the training data
training_data = []
for greeting in greetings:
    training_data.append((greeting, "greeting"))
for non_greeting in non_greetings:
    training_data.append((non_greeting, "non_greeting"))

# Convert to DataFrame for easier manipulation
df = pd.DataFrame(training_data, columns=["text", "label"])

# Shuffle the data
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Split into training and testing sets (80% train, 20% test)
train_size = int(0.8 * len(df))
train_df = df[:train_size]
test_df = df[train_size:]

# Create a pipeline with CountVectorizer and MultinomialNB
model = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])

# Train the model
model.fit(train_df['text'], train_df['label'])

# Evaluate on the test set
predictions = model.predict(test_df['text'])
accuracy = accuracy_score(test_df['label'], predictions)
print(f"Model accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(test_df['label'], predictions))

# Save the model data for future reference
def save_training_data():
    with open("greeting_training_data.txt", "w") as f:
        for text, label in zip(df['text'], df['label']):
            f.write(f"{text},{label}\n")
        
    print("Training data saved to greeting_training_data.txt")

# Function to use the trained model
def respond_to_message(message):
    # Predict if the message is a greeting
    prediction = model.predict([message])[0]
    
    if prediction == "greeting":
        # If it's a greeting, respond with a random greeting
        return random.choice(greeting_responses)
    else:
        # If it's not a greeting, return None or a default response
        return "I'm a simple greeting AI. I can only respond to greetings."

# Interactive demo
def interactive_demo():
    print("Greeting AI Demo (type 'exit' to quit)")
    print("--------------------------------------")
    
    while True:
        user_input = input("\nYou: ").lower().strip()
        
        if user_input == "exit":
            print("Goodbye!")
            break
            
        response = respond_to_message(user_input)
        print(f"AI: {response}")

# Example usage
if __name__ == "__main__":
    # Uncomment to save the training data
    # save_training_data()
    
    # Run the interactive demo
    print("\nDemo time!")
    interactive_demo()