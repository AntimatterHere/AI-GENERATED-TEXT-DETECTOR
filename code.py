pip install numpy pandas scikit-learn

import json
import pandas as pd

# Define the path to the uploaded JSONL file
file_path = '/content/all .jsonl'

# Initialize lists to store data
texts = []
labels = []

# Read the JSONL file line by line with error handling
with open(file_path, 'r', encoding='utf-8') as file:
    for line_number, line in enumerate(file, start=1):
        try:
            data = json.loads(line.strip())  # Strip any extra whitespace/newlines

            # Extract human-written text
            human_text = data.get('human_answers', [])
            for text in human_text:
                texts.append(text)
                labels.append(0)  # Label 0 for human-written

            # Extract AI-generated text
            ai_text = data.get('chatgpt_answers', [])
            for text in ai_text:
                texts.append(text)
                labels.append(1)  # Label 1 for AI-generated

        except json.JSONDecodeError as e:
            print(f"JSONDecodeError at line {line_number}: {e}")
            continue  # Skip problematic lines

# Create a DataFrame
df = pd.DataFrame({'text': texts, 'label': labels})

# Shuffle the DataFrame
df = df.sample(frac=1).reset_index(drop=True)

# Display the first few rows of the DataFrame
print("Sample Data:")
print(df.head())


import re  # Import the regular expressions module
from sklearn.model_selection import train_test_split

# Step 1: Text Cleaning Function
def clean_text(text):
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
    text = re.sub(r'[^A-Za-z\s]', '', text)     # Remove special characters
    text = text.lower()                         # Convert to lowercase
    return text

# Apply the text cleaning function
df['text'] = df['text'].apply(clean_text)

# Step 2: Split the data into features (X) and labels (y)
X = df['text']
y = df['label']

# Step 3: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the size of the training and testing sets
print(f"Training set size: {len(X_train)} samples")
print(f"Testing set size: {len(X_test)} samples")

# Display a sample of the cleaned text
print("Sample cleaned text from training data:")
print(X_train.head())



from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Step 1: Define maximum number of words and sequence length
max_words = 5000  # Maximum number of words to keep in the vocabulary
max_length = 100  # Maximum length of each input sequence

# Step 2: Initialize the Tokenizer
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train)  # Fit the tokenizer on the training data

# Step 3: Tokenize and pad the training and testing data
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Pad the sequences to ensure uniform length
X_train_pad = pad_sequences(X_train_seq, maxlen=max_length, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_length, padding='post')

# Display the shape of the padded data
print(f"Shape of training data: {X_train_pad.shape}")
print(f"Shape of testing data: {X_test_pad.shape}")

# Display a sample of the tokenized and padded data
print("Sample tokenized and padded sequence:")
print(X_train_pad[0])

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# Step 1: Define the LSTM Model Architecture
model = Sequential([
    Embedding(input_dim=5000, output_dim=128, input_length=100),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1, activation='sigmoid')  # Sigmoid activation for binary classification
])

# Step 2: Compile the Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 3: Train the Model
history = model.fit(X_train_pad, y_train, validation_data=(X_test_pad, y_test), epochs=5, batch_size=64)

# Display training summary
print("Model training completed.")


from sklearn.metrics import classification_report, accuracy_score

# Step 1: Make Predictions on the Test Data
y_pred = (model.predict(X_test_pad) > 0.5).astype(int)

# Step 2: Calculate Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"LSTM Model Accuracy: {accuracy:.2f}")

# Step 3: Generate a Classification Report
report = classification_report(y_test, y_pred, target_names=["Human-written", "AI-generated"])
print("Classification Report:\n", report)

# Define a function to clean, tokenize, and pad the input text
def clean_text(text):
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
    text = re.sub(r'[^A-Za-z\s]', '', text)     # Remove special characters
    text = text.lower()                         # Convert to lowercase
    return text

# Define a function to predict whether a given text is AI-generated or human-written
def predict_text(input_text, model, tokenizer, max_length=100):
    # Step 1: Clean the input text
    input_text = clean_text(input_text)

    # Step 2: Tokenize the input text
    input_seq = tokenizer.texts_to_sequences([input_text])

    # Step 3: Pad the sequence
    input_pad = pad_sequences(input_seq, maxlen=max_length, padding='post')

    # Step 4: Get the model's prediction
    prediction = model.predict(input_pad)[0][0]

    # Step 5: Return the result
    return "AI-generated" if prediction >= 0.5 else "Human-written"

# Loop for user input
while True:
    # Take input from the user
    user_input = input("Enter a text (or type 'exit' to quit): ")

    # Exit condition
    if user_input.lower() == 'exit':
        print("Exiting the AI Text Detector.")
        break

    # Get the prediction
    result = predict_text(user_input, model, tokenizer)

    # Display the result
    print(f"The input text is: {result}\n")
