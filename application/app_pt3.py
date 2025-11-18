import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

# -----------------------------
# PAGE SETUP
# -----------------------------
st.set_page_config(
    page_title="Apple WWDC Sentiment Dashboard",
    page_icon="🍏",
    layout="wide"
)

# -----------------------------
# LIGHT MODE & PASTEL THEME
# -----------------------------
st.markdown("""
<style>
body {
    background-color: #f9f6ff !important;
    color: #000000 !important;
}

.stMetric {
    background: linear-gradient(120deg, #f4e8ff, #fef9ff);
    padding: 10px 15px;
    border-radius: 10px;
    box-shadow: 0 2px 6px rgba(150, 100, 200, 0.1);
    color: #5e4b8b !important;
    font-weight: 600;
}

[data-testid="stMetricValue"] {
    color: #5e4b8b !important;
}

.stButton > button {
    background: linear-gradient(90deg, #c9a7eb, #e7c6ff);
    color: #000000;
    font-weight: bold;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 1.5rem;
    transition: 0.2s ease-in-out;
}
.stButton > button:hover {
    transform: scale(1.03);
    background: linear-gradient(90deg, #d8b7f0, #edd1ff);
}

.stDownloadButton > button {
    background: linear-gradient(90deg, #e0b0ff, #c5b3ff);
    color: #000000;
    border-radius: 10px;
    border: none;
    font-weight: bold;
}
.stDownloadButton > button:hover {
    transform: scale(1.03);
    background: linear-gradient(90deg, #e8c7ff, #d4c1ff);
}

footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# SIDEBAR NAVIGATION
# -----------------------------
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/f/fa/Apple_logo_black.svg", width=70)
st.sidebar.title("🍏 WWDC Dashboard")
page = st.sidebar.radio("Navigate", ["📊 Project Overview", "📥 Predict From CSV", "📝 Predict Single Text", "📈 Visualizations"])
st.sidebar.markdown("---")
st.sidebar.markdown("👩🏽‍💻 *By Ctrl Alt Elite — BSc IT (Data Science)*")

# -----------------------------
# Load model components once
# -----------------------------
@st.cache_resource
def load_model_components():
    MODEL_PATH = r"/Users/jeremypharell/Desktop/£/School/Year 3/ITDPA/model/models_saved/svm_sentiment_model.pkl"
    VECTORIZER_PATH = r"/Users/jeremypharell/Desktop/£/School/Year 3/ITDPA/model/models_saved/tfidf_vectorizer.pkl"
    SELECTOR_PATH = r"/Users/jeremypharell/Desktop/£/School/Year 3/ITDPA/model/models_saved/feature_selector.pkl"
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    selector = joblib.load(SELECTOR_PATH)
    return model, vectorizer, selector

model, vectorizer, selector = load_model_components()

# -----------------------------
# Text cleaning function (same as training)
# -----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def predict_sentiment(texts):
    cleaned = [clean_text(t) for t in texts]
    X_tfidf = vectorizer.transform(cleaned)
    X_selected = selector.transform(X_tfidf)
    preds = model.predict(X_selected)
    return preds

# -----------------------------
# PAGE 1: PROJECT OVERVIEW
# -----------------------------
if page == "📊 Project Overview":
    st.title("Apple WWDC Sentiment Analysis Dashboard 🍎")
    st.markdown("""
    Welcome to the **Apple WWDC Sentiment Analysis Dashboard**, a large-scale text classification project analyzing over  
    **4.1 million Reddit posts** related to Apple's WWDC.  
    The model uses **TF-IDF** + **balanced linear SVM** to predict **Positive / Neutral / Negative** sentiment.
    """)
    st.divider()

    st.subheader("📈 Model Performance Summary")
    st.markdown("**Overall Accuracy:** `81.15%`")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Negative F1", "0.59", help="Precision: 0.67 | Recall: 0.53")
    with col2:
        st.metric("Neutral F1", "0.83", help="Precision: 0.80 | Recall: 0.87")
    with col3:
        st.metric("Positive F1", "0.83", help="Precision: 0.87 | Recall: 0.80")

    # Confusion matrix
    cm = np.array([
        [205582, 173069, 8536],
        [95524, 1905648, 183688],
        [7187, 308862, 1232788]
    ])
    labels = ["Negative", "Neutral", "Positive"]

    st.markdown("### 🔍 Confusion Matrix")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", xticklabels=labels, yticklabels=labels)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    st.pyplot(fig)

    st.markdown("""
    **Interpretation:**  
    - Overall accuracy: **81%**  
    - Neutral sentiment dominates and is predicted most accurately  
    - Negative posts are the hardest to classify (recall = 0.53)  
    - Weighted SVM improved class balance
    """)

    st.divider()
    st.markdown("### 🧠 Model Configuration")
    st.code("""
SGDClassifier(
    loss='hinge',          # Linear SVM
    alpha=1e-5,            # Lower = more precise
    max_iter=5,
    tol=1e-3,
    random_state=42,
    class_weight=weights_dict
)
""", language="python")
    st.markdown("""
-   Training Data: 4,120,884 Reddit posts  
-   Vectorizer: TF-IDF (1–3 ngrams, 100k features)  
-   Output Classes: Negative / Neutral / Positive  
""")
    st.divider()
    st.markdown("💡 *Try uploading your own dataset on the next page to get predictions!*")

# -----------------------------
# PAGE 2: PREDICT FROM CSV
# -----------------------------
if page == "📥 Predict From CSV":
    st.title("📥 Predict Sentiment From Your CSV")

    uploaded_file = st.file_uploader("Upload a CSV file with a 'text' column", type=["csv"])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            if "text" not in df.columns:
                st.error("❌ CSV must include a column named 'text'.")
            else:
                st.success(f"✅ Loaded {df.shape[0]} rows successfully!")

                with st.spinner("🧠 Predicting sentiments..."):
                    df["clean_text"] = df["text"].apply(clean_text)
                    X_tfidf = vectorizer.transform(df["clean_text"])
                    X_selected = selector.transform(X_tfidf)
                    df["predicted_sentiment"] = model.predict(X_selected)

                st.success("🎉 Predictions complete!")
                st.markdown("### 🔎 Sample Predictions")
                st.dataframe(df[["text", "predicted_sentiment"]].head(10))

                st.markdown("### 📊 Sentiment Distribution")
                counts = df["predicted_sentiment"].value_counts()
                st.bar_chart(counts)

                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download Predictions", csv, "predicted_sentiments.csv", "text/csv")

        except Exception as e:
            st.error(f"⚠️ Error reading file: {e}")

# -----------------------------
# PAGE 3: PREDICT SINGLE TEXT
# -----------------------------
if page == "📝 Predict Single Text":
    st.title("📝 Predict Sentiment From Text Input")

    user_input = st.text_area("Enter your text here:", height=150)

    if st.button("Predict Sentiment"):
        if user_input.strip():
            cleaned_text = clean_text(user_input)
            X_tfidf = vectorizer.transform([cleaned_text])
            X_selected = selector.transform(X_tfidf)
            prediction = model.predict(X_selected)[0]
            st.success(f"Predicted sentiment: **{prediction}**")
        else:
            st.warning("Please enter some text to predict.")

# -----------------------------
# PAGE 4: VISUALIZATIONS
# -----------------------------
if page == "📈 Visualizations":
    st.title("📊 Model & Dataset Visualizations")

    data_path = r"/Users/jeremypharell/Desktop/£/School/Year 3/ITDPA/csv's/cleaned_combine_datasets.csv"
    df = pd.read_csv(data_path, usecols=['combined_text', 'clean_text', 'label'])
    df = df.dropna(subset=['clean_text', 'label'])

    labels = ['Negative', 'Neutral', 'Positive']
    sentiment_counts = df['label'].value_counts()

    conf_matrix = np.array([
        [205582, 173069, 8536],
        [95524, 1905648, 183688],
        [7187, 308862, 1232788]
    ])
    accuracy = 0.8115
    actual_counts = [conf_matrix[i].sum() for i in range(3)]
    predicted_counts = [conf_matrix[:, i].sum() for i in range(3)]

    # FIGURE 1: Dataset Info + Sentiment Distribution
    fig1, axs1 = plt.subplots(2, 1, figsize=(10, 8))
    plt.subplots_adjust(hspace=0.5)

    axs1[0].axis('off')
    info_text = f"Dataset size: {len(df)} lines\nImportant keywords: WWDC, Apple, Swift, Siri, iOS, Mac, AI"
    axs1[0].text(0.5, 0.5, info_text, fontsize=14, ha='center', va='center', fontweight='bold', color="#6C5B7B")

    sns.countplot(x='label', data=df, order=labels, palette='pastel', ax=axs1[1])
    axs1[1].set_title("Sentiment Distribution", fontsize=14, fontweight='bold')
    axs1[1].set_ylabel("Number of Posts")

    st.pyplot(fig1)

    # FIGURE 2: Sentiment Pie + Confusion Matrix
    fig2, axs2 = plt.subplots(2, 1, figsize=(10, 10))
    plt.subplots_adjust(hspace=0.5)

    axs2[0].pie(sentiment_counts[labels], labels=labels, autopct='%1.1f%%',
                colors=sns.color_palette('pastel'), startangle=90, textprops={'fontsize': 12})
    axs2[0].set_title("Sentiment Proportions", fontsize=14, fontweight='bold')

    pastel_cmap = ListedColormap(sns.color_palette("pastel", 8).as_hex())
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap=pastel_cmap,
                xticklabels=labels, yticklabels=labels, linewidths=0.5, cbar=False, ax=axs2[1])
    axs2[1].set_xlabel("Predicted Label")
    axs2[1].set_ylabel("True Label")
    axs2[1].set_title("Confusion Matrix", fontsize=14, fontweight='bold')

    st.pyplot(fig2)

    # FIGURE 3: Actual vs Predicted + Accuracy
    fig3, axs3 = plt.subplots(2, 1, figsize=(10, 10))
    plt.subplots_adjust(hspace=0.5)

    dist_df = pd.DataFrame({'Sentiment': labels, 'Actual': actual_counts, 'Predicted': predicted_counts})
    dist_df_melted = dist_df.melt(id_vars='Sentiment', var_name='Type', value_name='Count')
    sns.barplot(x='Sentiment', y='Count', hue='Type', data=dist_df_melted, palette='pastel', ax=axs3[0])
    axs3[0].set_title("Actual vs Predicted Distribution", fontsize=14, fontweight='bold')
    axs3[0].set_ylabel("Number of Posts")

    axs3[1].barh(['Accuracy'], [accuracy], color="#A7C7E7", edgecolor='white')
    axs3[1].barh(['Accuracy'], [1 - accuracy], left=[accuracy], color="#F0F0F0", edgecolor='white')
    axs3[1].text(accuracy / 2, 0, f"{accuracy * 100:.1f}%", va='center', ha='center', fontsize=14, fontweight='bold', color='black')
    axs3[1].set_xlim(0, 1)
    axs3[1].set_xticks([])
    axs3[1].set_yticks([])
    axs3[1].set_title("Overall Model Accuracy", fontsize=14, fontweight='bold')

    st.pyplot(fig3)


