import os, re
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

def load_data (directory):
    texts = []
    labels = []
    # 1 = Posiive, 0 = Negative
    for label_type in ["pos", "neg"]:
        dir_path = os.path.join(directory, label_type)
        if label_type == "pos":
            label = 1
        else:
            label = 0
        for fname in os.listdir(dir_path):
            if fname.endswith(".txt"):
                with open(os.path.join(dir_path, fname), "r", encoding="utf-8") as f:
                    texts.append(f.read())
                labels.append(label)
    return texts, labels

train_texts, train_labels = load_data("extracted_files/aclImdb/train")
test_texts, test_labels = load_data("extracted_files/aclImdb/test")

# Split original TRAIN data into train/val (80-20)
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_texts, train_labels,
    test_size=0.2,
    random_state=42
) 

# Prepocessing (Removing HTML tags and only including alphanumericals characters in text)
def preprocess_text(text):
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Lowercase and remove non-alphanumeric
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    return text

# Clean texts in all splits
train_texts = [preprocess_text(text) for text in train_texts]
val_texts = [preprocess_text(text) for text in val_texts]
test_texts = [preprocess_text(text) for text in test_texts]

# Tokenizer only sees training data now
tokenizer = Tokenizer(num_words=10000)  # Keep top 10k words
tokenizer.fit_on_texts(train_texts)     # Learn vocabulary from training data (This fits only on training data to keep bias 0)

# Convert texts to sequences (Converts text to integers)
train_sequences = tokenizer.texts_to_sequences(train_texts)     
val_sequences = tokenizer.texts_to_sequences(val_texts)         
test_sequences = tokenizer.texts_to_sequences(test_texts)       

# Pad sequences to uniform length
maxlen = 200  # Truncate/pad to 200 tokens for all splits
X_train = pad_sequences(train_sequences, maxlen=maxlen)
X_val = pad_sequences(val_sequences, maxlen=maxlen)
X_test = pad_sequences(test_sequences, maxlen=maxlen)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

model = Sequential([
    Embedding(input_dim=10000, output_dim=128, input_length=maxlen),        # Converts integer toxens to dense vectors
    LSTM(64, dropout=0.2, recurrent_dropout=0.2),                           # LSTM with regularization
    Dense(1, activation="sigmoid")                                          # Binary classification
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    X_train, train_labels,
    epochs=5,
    batch_size=64,
    validation_data=(X_val, val_labels)
)

# Evaluate on test set
test_loss, test_acc = model.evaluate(X_test, test_labels)
print(f"Final Test Accuracy: {test_acc:.4f}")

# Preprocesses input text, converts to tokens and classifies sentiment
def predict_sentiment(text):
    text = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=maxlen)
    prediction = model.predict(padded)
    return "Positive" if prediction > 0.5 else "Negative", prediction[0][0]

# Example
sample_text = "This movie was an amazing masterpiece that I'll never forget!"
sentiment, confidence = predict_sentiment(sample_text)
print(f"Sentiment: {sentiment} (Confidence: {confidence:.2f})")