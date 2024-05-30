import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import tkinter as tk
from tkinter import StringVar, Label, ttk

# Load dataset
data = pd.read_csv('spam.csv', encoding='latin-1')
data = data[['Category', 'Message']]
data['Category'] = data['Category'].map({'ham': 0, 'spam': 1})

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = ''.join([c for c in text if c.isalnum() or c.isspace()])
    return text

data['Message'] = data['Message'].apply(preprocess_text)

# Split the data
X = data['Message']
y = data['Category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature extraction
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Model training
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Model evaluation
y_pred = model.predict(X_test_tfidf)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(classification_report(y_test, y_pred))

# GUI development
def classify_message(*args):
    message = message_var.get()
    if message:
        message = preprocess_text(message)
        message_tfidf = vectorizer.transform([message])
        prediction = model.predict(message_tfidf)[0]
        result = 'Spam' if prediction == 1 else 'Ham'
        result_var.set(f"The message is classified as: {result}")
    else:
        result_var.set("")

# Create GUI
root = tk.Tk()
root.title("Spam Detector")
root.geometry("600x300")
root.resizable(False, False)

style = ttk.Style()
style.configure("TLabel", font=("Helvetica", 12))
style.configure("TEntry", font=("Helvetica", 12))
style.configure("TFrame", padding="10")

main_frame = ttk.Frame(root)
main_frame.pack(fill="both", expand=True)

message_var = StringVar()
message_var.trace_add('write', classify_message)

result_var = StringVar()

ttk.Label(main_frame, text="Enter your message:").pack(pady=10)
entry = ttk.Entry(main_frame, width=70, textvariable=message_var)
entry.pack(pady=10)

label_result = ttk.Label(main_frame, textvariable=result_var, foreground="blue")
label_result.pack(pady=10)

root.mainloop()
