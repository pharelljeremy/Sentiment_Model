import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from gensim import corpora, models
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob
import re
from collections import Counter
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# --- Parameters ---
INPUT_CSV = "/Users/jeremypharell/Desktop/£/School/Year 3/ITDPA/topic_modelling/csv's/cleaned_combine_datasets.csv"
CHUNKSIZE = 100_000
NUM_TOPICS = 10
NUM_WORDS = 10
OUTPUT_DIR = "/Users/jeremypharell/Desktop/£/School/Year 3/ITDPA/topic_modelling/images"
TOPIC_ASSIGNMENTS_CSV = "topic_assignments.csv"
SENTIMENT_ASSIGNMENTS_CSV = "sentiment_assignments.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Text preprocessing ---
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    return tokens

# --- Sentiment analysis with parallelization ---
def get_sentiment(text):
    try:
        return TextBlob(text).sentiment.polarity
    except Exception:
        return 0.0

def parallel_sentiment(texts, max_workers=8):
    sentiments = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(get_sentiment, text): idx for idx, text in enumerate(texts)}
        sentiments = [0.0] * len(texts)
        for future in as_completed(futures):
            idx = futures[future]
            try:
                sentiments[idx] = future.result()
            except Exception:
                sentiments[idx] = 0.0
    return sentiments

# --- Progress helper ---
def print_progress(current, total, start_time):
    elapsed = time.time() - start_time
    progress = current / total
    eta = (elapsed / progress) * (1 - progress) if progress > 0 else float('inf')
    print(f"Processed {current}/{total} rows ({progress*100:.2f}%), ETA: {eta/60:.2f} min")

# --- Step 1: Count total rows (excluding header) ---
print("Counting total rows...")
with open(INPUT_CSV, 'r', encoding='utf-8') as f:
    total_lines = sum(1 for _ in f) - 1
print(f"Total rows: {total_lines}")

# --- Step 2: Build dictionary incrementally ---
print("Building dictionary...")
dictionary = corpora.Dictionary()
rows_processed = 0
start_time = time.time()

for chunk in pd.read_csv(INPUT_CSV, chunksize=CHUNKSIZE, usecols=['clean_text']):
    texts = chunk['clean_text'].dropna().astype(str).apply(preprocess).tolist()
    dictionary.add_documents(texts)
    rows_processed += len(chunk)
    print_progress(rows_processed, total_lines, start_time)

dictionary.filter_extremes(no_below=5, no_above=0.5)
print(f"Dictionary size: {len(dictionary)} tokens")

# --- Step 3: Train LDA incrementally ---
print("Training LDA model incrementally...")
lda_model = models.LdaModel(id2word=dictionary, num_topics=NUM_TOPICS, passes=1, update_every=1, chunksize=CHUNKSIZE, random_state=42)

rows_processed = 0
start_time = time.time()

for i, chunk in enumerate(pd.read_csv(INPUT_CSV, chunksize=CHUNKSIZE, usecols=['clean_text'])):
    texts = chunk['clean_text'].dropna().astype(str).apply(preprocess).tolist()
    corpus = [dictionary.doc2bow(text) for text in texts]
    lda_model.update(corpus)
    rows_processed += len(chunk)
    print_progress(rows_processed, total_lines, start_time)

print("LDA training complete.")

# --- Step 4: Assign dominant topic and sentiment per chunk and save ---
print("Assigning topics and sentiment and saving results incrementally...")
if os.path.exists(TOPIC_ASSIGNMENTS_CSV):
    os.remove(TOPIC_ASSIGNMENTS_CSV)
if os.path.exists(SENTIMENT_ASSIGNMENTS_CSV):
    os.remove(SENTIMENT_ASSIGNMENTS_CSV)

rows_processed = 0
start_time = time.time()

for chunk in pd.read_csv(INPUT_CSV, chunksize=CHUNKSIZE, usecols=['clean_text']):
    texts = chunk['clean_text'].dropna().astype(str).tolist()
    preprocessed_texts = [preprocess(text) for text in texts]
    corpus = [dictionary.doc2bow(text) for text in preprocessed_texts]

    dominant_topics = []
    topic_probs = []
    for bow in corpus:
        topics = lda_model.get_document_topics(bow)
        if topics:
            topic_id, topic_prob = max(topics, key=lambda x: x[1])
            dominant_topics.append(topic_id)
            topic_probs.append(topic_prob)
        else:
            dominant_topics.append(-1)
            topic_probs.append(0)

    sentiments = parallel_sentiment(texts, max_workers=4)  # Adjust workers if needed

    df_out = pd.DataFrame({
        'dominant_topic': dominant_topics,
        'topic_prob': topic_probs,
        'sentiment': sentiments
    })

    header_topic = not os.path.exists(TOPIC_ASSIGNMENTS_CSV)
    header_sentiment = not os.path.exists(SENTIMENT_ASSIGNMENTS_CSV)

    df_out[['dominant_topic', 'topic_prob']].to_csv(TOPIC_ASSIGNMENTS_CSV, mode='a', header=header_topic, index=False)
    df_out[['sentiment']].to_csv(SENTIMENT_ASSIGNMENTS_CSV, mode='a', header=header_sentiment, index=False)

    rows_processed += len(chunk)
    print_progress(rows_processed, total_lines, start_time)

# --- Step 5: Load assignments for plotting ---
print("Loading assignments for plotting...")
topics_df = pd.read_csv(TOPIC_ASSIGNMENTS_CSV)
sentiment_df = pd.read_csv(SENTIMENT_ASSIGNMENTS_CSV)

# --- Step 6: Plot and save graphs ---
def save_plot(fig, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved plot: {path}")

# Topic distribution
fig, ax = plt.subplots(figsize=(10,6))
sns.countplot(x='dominant_topic', data=topics_df, palette='viridis', ax=ax)
ax.set_title("Document Count per Dominant Topic")
ax.set_xlabel("Topic")
ax.set_ylabel("Number of Documents")
save_plot(fig, "topic_distribution.png")

# Sentiment distribution
fig, ax = plt.subplots(figsize=(10,6))
sns.histplot(sentiment_df['sentiment'], bins=50, kde=True, color='purple', ax=ax)
ax.set_title("Sentiment Polarity Distribution")
ax.set_xlabel("Sentiment Polarity (-1 negative, +1 positive)")
ax.set_ylabel("Frequency")
save_plot(fig, "sentiment_distribution.png")

# Sentiment per topic boxplot
merged_df = pd.concat([topics_df, sentiment_df], axis=1)
fig, ax = plt.subplots(figsize=(12,6))
sns.boxplot(x='dominant_topic', y='sentiment', data=merged_df, palette='viridis', ax=ax)
ax.set_title("Sentiment Distribution per Topic")
ax.set_xlabel("Topic")
ax.set_ylabel("Sentiment Polarity")
save_plot(fig, "sentiment_per_topic_boxplot.png")

# Average sentiment per topic bar chart
avg_sentiment = merged_df.groupby('dominant_topic')['sentiment'].mean().reset_index()
fig, ax = plt.subplots(figsize=(10,6))
sns.barplot(x='dominant_topic', y='sentiment', data=avg_sentiment, palette='viridis', ax=ax)
ax.set_title("Average Sentiment Polarity per Topic")
ax.set_xlabel("Topic")
ax.set_ylabel("Average Sentiment Polarity")
save_plot(fig, "average_sentiment_per_topic.png")

# Word frequency overall (from dictionary)
word_freq = {dictionary[id]: freq for id, freq in dictionary.dfs.items()}
most_common_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20]
words, counts = zip(*most_common_words)
fig, ax = plt.subplots(figsize=(12,6))
sns.barplot(x=list(counts), y=list(words), palette='viridis', ax=ax)
ax.set_title("Top 20 Most Frequent Words Overall")
ax.set_xlabel("Frequency")
ax.set_ylabel("Word")
save_plot(fig, "top_words_overall.png")

# Word clouds per topic
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.flatten()
for i, ax in enumerate(axes):
    words_probs = lda_model.show_topic(i, topn=NUM_WORDS)
    freq_dict = {w: p for w, p in words_probs}
    wc = WordCloud(background_color='white', colormap='viridis', width=400, height=300)
    wc.generate_from_frequencies(freq_dict)
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(f"Topic {i}")
plt.suptitle("Word Clouds for Topics", fontsize=18)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(OUTPUT_DIR, "wordclouds_per_topic.png"), bbox_inches='tight')
plt.close()
print(f"Saved plot: {os.path.join(OUTPUT_DIR, 'wordclouds_per_topic.png')}")

print("All done!")
