import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report
import joblib
import matplotlib.pyplot as plt

# Loading data from CSV
df = pd.read_csv('spam.csv', encoding='latin-1')

# Division into message and label (spam or not spam)
X = df['text']  # text message
y = df['label']  # label spam

# Converting labels into numbers (not spam -> 0, spam -> 1)
y = y.apply(lambda x: 1 if x == 'spam' else 0)

# Splitting to learning and test model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF Vectoring text data
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Learning models
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Saving models
joblib.dump(model, 'spam_classifier.pkl')
joblib.dump(tfidf_vectorizer, 'vectorizer.pkl')

# Model evaluation on a test sample
y_pred = model.predict(X_test_tfidf)

# Output of model evaluation metrics
print('\nConfusion Matrix:')
print(confusion_matrix(y_test, y_pred))

print('\nClassification Report:')
print(classification_report(y_test, y_pred))

# Calculate precision, recall, and f1-score
report = classification_report(y_test, y_pred, output_dict=True)
metrics = {'precision': [], 'recall': [], 'f1-score': []}
metrics['precision'].append(report['0']['precision'])
metrics['recall'].append(report['0']['recall'])
metrics['f1-score'].append(report['0']['f1-score'])
metrics['precision'].append(report['1']['precision'])
metrics['recall'].append(report['1']['recall'])
metrics['f1-score'].append(report['1']['f1-score'])

# Plotting results
labels = ['Не спам', 'Cпам']
metrics_names = list(metrics.keys())
metrics_values = list(metrics.values())

x = range(len(labels))
width = 0.2

fig, ax = plt.subplots(figsize=(8, 6))
for i, metric in enumerate(metrics_names):
    ax.bar([pos + width * i for pos in x], metrics_values[i], width, label=metric)

ax.set_ylabel('Счет')
ax.set_title('Метрики по классам')
ax.set_xticks([pos + width for pos in x])
ax.set_xticklabels(labels)
ax.legend(loc='best')

plt.ylim(0, 1.2)
plt.grid(True)
plt.tight_layout()
plt.show()
