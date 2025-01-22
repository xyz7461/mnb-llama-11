import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold
from transformers import AutoTokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

import os
import pandas as pd

def load_documents_from_files(root_dir):
    documents = []
    labels = []
    
    for file_name in os.listdir(root_dir):
        file_path = os.path.join(root_dir, file_name)
        if file_name.endswith(".txt") and os.path.isfile(file_path):
            label = os.path.splitext(file_name)[0] 
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                questions_answers = content.split('\n')
                
                for i in range(0, len(questions_answers) - 1, 2):
                    question = questions_answers[i].strip()
                    answer = questions_answers[i + 1].strip()
                    if question and answer: 
                        documents.append(f"{question} {answer}")
                        labels.append(label)
    
    return pd.DataFrame({'text': documents, 'label': labels})


dataset_df = load_documents_from_files('/Users/kenziecottle/VS Code Projects/Project 4 - On Classification/train-easy-full')

# Prepare the features and labels for Naive Bayes
X = dataset_df['text']
y = dataset_df['label']

# Vectorize the text data for Naive Bayes
vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X)

# Encode labels for transformer model
label_encoder = LabelEncoder()
dataset_df['label'] = label_encoder.fit_transform(dataset_df['label'])

# Tokenizer for transformer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

# Perform 9-fold cross-validation
kf = KFold(n_splits=9, shuffle=True, random_state=42)
nb_accuracies = []
transformer_accuracies = []

for train_index, test_index in kf.split(X_vec):
    # Split data for Naive Bayes model
    X_train, X_test = X_vec[train_index], X_vec[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Train and evaluate Naive Bayes model
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    y_pred = nb_model.predict(X_test)
    nb_accuracy = accuracy_score(y_test, y_pred)
    nb_accuracies.append(nb_accuracy)
    print(f"Naive Bayes Fold Accuracy: {nb_accuracy}")

# Print cross-validated results for both models
print("9-Fold Cross-Validation Results:")
print(f"Naive Bayes Mean Accuracy: {sum(nb_accuracies)/len(nb_accuracies)}")

