import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.utils.class_weight import compute_class_weight
from joblib import dump, load
from scipy import sparse
from tqdm import tqdm

# Paths and settings
base_dir = r"/Users/jeremypharell/Desktop/£/School/Year 3/ITDPA/model/models_saved"
labeled_csv_path = os.path.join(base_dir, "labeled_dataset_vader.csv")
vectorizer_path = os.path.join(base_dir, "tfidf_vectorizer.pkl")
selector_path = os.path.join(base_dir, "feature_selector.pkl")
model_path = os.path.join(base_dir, "svm_sentiment_model.pkl")
tfidf_chunks_dir = os.path.join(base_dir, "tfidf_chunks")

chunk_size = 10000  # Adjust based on your RAM

os.makedirs(tfidf_chunks_dir, exist_ok=True)

# Step 1: Fit TF-IDF vectorizer on a sample chunk and save
print("Fitting TF-IDF vectorizer on sample chunk...")
sample_reader = pd.read_csv(labeled_csv_path, chunksize=chunk_size, low_memory=True)
sample_chunk = next(sample_reader)
sample_chunk['clean_text'] = sample_chunk['clean_text'].fillna('')
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True, stop_words='english')
vectorizer.fit(sample_chunk['clean_text'])
dump(vectorizer, vectorizer_path)
print(f"Vectorizer saved to {vectorizer_path}")

# Step 2: Transform full data in chunks and save sparse matrices
print("Transforming text data in chunks and saving TF-IDF features...")
reader = pd.read_csv(labeled_csv_path, chunksize=chunk_size, low_memory=True)
chunk_files = []
for i, chunk in enumerate(tqdm(reader)):
    chunk['clean_text'] = chunk['clean_text'].fillna('')
    texts = chunk['clean_text']
    X_tfidf = vectorizer.transform(texts)
    chunk_file = os.path.join(tfidf_chunks_dir, f"chunk_{i}.npz")
    sparse.save_npz(chunk_file, X_tfidf)
    chunk_files.append((chunk_file, chunk['label'].values))
print(f"Saved {len(chunk_files)} TF-IDF chunks.")

# Step 3: Fit feature selector on the sample chunk and save
print("Fitting feature selector on sample chunk...")
X_sample = vectorizer.transform(sample_chunk['clean_text'])
selector = SelectKBest(chi2, k=2000)
selector.fit(X_sample, sample_chunk['label'])
dump(selector, selector_path)
print(f"Feature selector saved to {selector_path}")

# Step 4: Compute class weights using all labels from CSV in chunks
print("Computing class weights...")
labels = []
reader = pd.read_csv(labeled_csv_path, chunksize=chunk_size, usecols=['label'], low_memory=True)
for chunk in tqdm(reader):
    labels.extend(chunk['label'].values)
labels = np.array(labels)
classes = np.unique(labels)
class_weights = compute_class_weight('balanced', classes=classes, y=labels)
class_weight_dict = dict(zip(classes, class_weights))
print(f"Class weights: {class_weight_dict}")

# Step 5: Initialize SGDClassifier
clf = SGDClassifier(loss='hinge', max_iter=1, tol=None, class_weight=class_weight_dict,
                    random_state=42, learning_rate='optimal')

# Step 6: Incremental training on TF-IDF chunks
print("Starting incremental training on TF-IDF chunks...")
for i, (chunk_file, y_chunk) in enumerate(tqdm(chunk_files)):
    X_chunk = sparse.load_npz(chunk_file)
    X_selected = selector.transform(X_chunk)
    if i == 0:
        clf.partial_fit(X_selected, y_chunk, classes=classes)
    else:
        clf.partial_fit(X_selected, y_chunk)

# Step 7: Save final model
dump(clf, model_path)
print(f"Training complete! Model saved to {model_path}")

