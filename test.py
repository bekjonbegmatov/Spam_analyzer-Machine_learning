import joblib
from sklearn.feature_extraction.text import CountVectorizer

# Loading model and vectorizer 
model = joblib.load('spam_classifier.pkl')
vectorizer = joblib.load('vectorizer.pkl')  

# Text examples
new_texts = [
    "Hello, I'm in Ulan-Ude now. See you soon :)",
    "Get free treal in our new Online platform, only for 20$",
    "Hello Masaridin, I'm in Ulan-Ude now. How are you ?",
    "Register for our seminar on digital marketing",
    "Congratulation you are winner"
]

# Converting texts to a numeric matrix using save vectorizer
new_texts_transformed = vectorizer.transform(new_texts)

# Prediction using a loaded model
new_predictions = model.predict(new_texts_transformed)

# Print results
for text, prediction in zip(new_texts, new_predictions):
    print(f"Text: \"{text}\" -> Prediction: {'spam' if prediction == 1 else 'not spam'}")
