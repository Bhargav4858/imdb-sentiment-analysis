
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import re
import html
import nltk
import os

from gensim.models import FastText
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="IMDb Sentiment Analysis",
    page_icon="🎬",
    layout="wide"
)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>
.main-title {
    font-size: 38px;
    font-weight: bold;
    text-align: center;
    color: #4F8BF9;
}
.sub-title {
    text-align: center;
    color: gray;
    font-size: 17px;
}
.result-box {
    padding: 25px;
    border-radius: 12px;
    text-align: center;
    font-size: 25px;
    font-weight: bold;
}
.positive {
    background-color: #d4edda;
    color: #155724;
}
.negative {
    background-color: #f8d7da;
    color: #721c24;
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# FILE PATHS
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FASTTEXT_PATH = os.path.join(
    BASE_DIR, "FastText_model.model"
)

ANN_PATH = os.path.join(
    BASE_DIR,
    "IMDB_Sentiment_Analysis_(Fast_Text&ANN).keras"
)

ENCODER_PATH = os.path.join(
    BASE_DIR,
    "imdb_label_encoder.pkl"
)


# --------------------------------------------------
# NLTK DOWNLOAD
# --------------------------------------------------

@st.cache_resource
def download_nltk_resources():

    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)

    return True


download_nltk_resources()


# --------------------------------------------------
# LOAD TRAINED MODELS
# --------------------------------------------------

@st.cache_resource
def load_assets():

    fasttext_model = FastText.load(
        FASTTEXT_PATH,
        mmap="r"
    )

    ann_model = load_model(
        ANN_PATH,
        compile=False
    )

    label_encoder = joblib.load(
        ENCODER_PATH
    )

    return fasttext_model, ann_model, label_encoder


try:

    with st.spinner("Loading trained models..."):

        fasttext_model, ann_model, label_encoder = load_assets()

except Exception as e:

    st.error("Unable to load models.")

    st.exception(e)

    st.stop()


# --------------------------------------------------
# TEXT PREPROCESSING
# --------------------------------------------------

stop_words = set(stopwords.words("english"))

# Preserve negation words
stop_words -= {"no", "nor", "not"}

lemmatizer = WordNetLemmatizer()


def preprocess_text(text):

    text = html.unescape(str(text))

    text = re.sub(r"<[^>]+>", " ", text)

    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    text = text.lower()

    text = re.sub(r"\bwon't\b", "will not", text)
    text = re.sub(r"\bcan't\b", "cannot", text)
    text = re.sub(r"\bshan't\b", "shall not", text)
    text = re.sub(r"\bain't\b", "is not", text)
    text = re.sub(r"n't\b", " not", text)
    text = re.sub(r"'re\b", " are", text)
    text = re.sub(r"'ve\b", " have", text)
    text = re.sub(r"'ll\b", " will", text)
    text = re.sub(r"'m\b", " am", text)
    text = re.sub(r"'d\b", " would", text)

    text = re.sub(
        r"[^a-z\s]",
        " ",
        text
    )

    tokens = text.split()

    tokens = [
        word for word in tokens
        if word not in stop_words
    ]

    tokens = [
        lemmatizer.lemmatize(word, pos="v")
        for word in tokens
    ]

    return tokens


# --------------------------------------------------
# FASTTEXT WORD EMBEDDINGS
# --------------------------------------------------

def get_review_vector(tokens):

    vectors = []

    for word in tokens:

        vector = fasttext_model.wv[word]

        vectors.append(vector)

    if len(vectors) == 0:

        return np.zeros(100, dtype=np.float32)

    return np.mean(
        vectors,
        axis=0
    ).astype(np.float32)


# --------------------------------------------------
# SENTIMENT PREDICTION
# --------------------------------------------------

def predict_sentiment(review):

    tokens = preprocess_text(review)

    if len(tokens) == 0:

        raise ValueError(
            "Please enter a valid movie review."
        )

    review_vector = get_review_vector(tokens)

    review_vector = review_vector.reshape(1, -1)

    prediction = ann_model.predict(
        review_vector,
        verbose=0
    )

    probability = float(
        np.asarray(prediction).reshape(-1)[0]
    )

    probability = np.clip(probability, 0, 1)

    if probability >= 0.5:

        sentiment = label_encoder.classes_[1]

        confidence = probability

    else:

        sentiment = label_encoder.classes_[0]

        confidence = 1 - probability

    return sentiment, confidence, probability, tokens


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.title("🎬 IMDb Analyzer")

    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "Home",
            "Sentiment Prediction",
            "Batch Prediction",
            "Model Information"
        ]
    )

    st.markdown("---")

    st.caption(
        "FastText + Artificial Neural Network"
    )


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

if page == "Home":

    st.markdown(
        '<div class="main-title">IMDb Sentiment Analysis</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sub-title">Movie Review Classification using FastText + ANN</div>',
        unsafe_allow_html=True
    )

    st.write("")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Embedding Model",
        "FastText"
    )

    col2.metric(
        "Classifier",
        "ANN"
    )

    col3.metric(
        "Output Classes",
        "2"
    )

    st.markdown("---")

    st.subheader("Project Overview")

    st.write("""
    This application predicts whether an IMDb movie review
    expresses positive or negative sentiment.

    It uses FastText word embeddings to convert textual
    reviews into numerical vectors and an Artificial
    Neural Network to classify the sentiment.
    """)

    st.subheader("Workflow")

    st.markdown("""
    1. Enter a movie review.
    2. Perform text preprocessing.
    3. Generate FastText word embeddings.
    4. Calculate the average word vector.
    5. Pass the vector into the trained ANN model.
    6. Display the sentiment and confidence score.
    """)


# --------------------------------------------------
# SINGLE REVIEW PREDICTION
# --------------------------------------------------

elif page == "Sentiment Prediction":

    st.header("🎯 Sentiment Prediction")

    st.write(
        "Enter your IMDb movie review below."
    )

    review = st.text_area(
        "Movie Review",
        height=180,
        placeholder="Enter your movie review..."
    )

    if st.button(
        "Predict Sentiment",
        type="primary",
        use_container_width=True
    ):

        if review.strip() == "":

            st.warning(
                "Please enter a review."
            )

        else:

            try:

                sentiment, confidence, probability, tokens = (
                    predict_sentiment(review)
                )

                st.markdown("---")

                if str(sentiment).lower() == "positive":

                    st.markdown(
                        f"""
                        <div class="result-box positive">
                            😊 Positive Review
                            <br>
                            Confidence: {confidence:.2%}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                else:

                    st.markdown(
                        f"""
                        <div class="result-box negative">
                            ☹️ Negative Review
                            <br>
                            Confidence: {confidence:.2%}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                st.subheader("Prediction Probabilities")

                col1, col2 = st.columns(2)

                col1.metric(
                    "Positive Probability",
                    f"{probability:.2%}"
                )

                col2.metric(
                    "Negative Probability",
                    f"{1-probability:.2%}"
                )

                st.progress(confidence)

                with st.expander("View processed text"):

                    st.write(" ".join(tokens))

            except Exception as e:

                st.error(str(e))


# --------------------------------------------------
# BATCH PREDICTION
# --------------------------------------------------

elif page == "Batch Prediction":

    st.header("📊 Batch Sentiment Analysis")

    st.write(
        "Upload a CSV file containing a column named 'review'."
    )

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

        if "review" not in df.columns:

            st.error(
                "CSV must contain a 'review' column."
            )

        else:

            st.write(df.head())

            if st.button(
                "Analyze All Reviews",
                type="primary"
            ):

                results = []

                progress = st.progress(0)

                for i, review in enumerate(
                    df["review"].fillna("")
                ):

                    try:

                        sentiment, confidence, probability, _ = (
                            predict_sentiment(str(review))
                        )

                        results.append({
                            "Review": review,
                            "Sentiment": sentiment,
                            "Confidence": confidence,
                            "Positive Probability": probability,
                            "Negative Probability": 1-probability
                        })

                    except ValueError:

                        results.append({
                            "Review": review,
                            "Sentiment": "Invalid",
                            "Confidence": 0,
                            "Positive Probability": 0,
                            "Negative Probability": 0
                        })

                    progress.progress(
                        (i + 1) / len(df)
                    )

                result_df = pd.DataFrame(results)

                st.success(
                    "Batch prediction completed!"
                )

                st.dataframe(
                    result_df,
                    use_container_width=True
                )

                st.subheader(
                    "Sentiment Distribution"
                )

                counts = (
                    result_df["Sentiment"]
                    .value_counts()
                )

                st.bar_chart(counts)

                csv = result_df.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    "Download Results",
                    csv,
                    "sentiment_results.csv",
                    "text/csv"
                )


# --------------------------------------------------
# MODEL INFORMATION
# --------------------------------------------------

elif page == "Model Information":

    st.header("🧠 Model Information")

    st.subheader("FastText Configuration")

    st.write("""
    - Vector Size: 100
    - Window Size: 5
    - Minimum Word Count: 1
    - Training Epochs: 20
    """)

    st.subheader("ANN Architecture")

    architecture = pd.DataFrame({
        "Layer": [
            "Input",
            "Dense 1",
            "Dense 2",
            "Output"
        ],
        "Neurons": [
            100,
            32,
            16,
            1
        ],
        "Activation": [
            "-",
            "ReLU",
            "ReLU",
            "Sigmoid"
        ]
    })

    st.table(architecture)

    st.subheader("Model Details")

    st.write("""
    - Optimizer: Adam
    - Loss Function: Binary Crossentropy
    - Batch Size: 32
    - Maximum Epochs: 20
    """)

    st.info(
        "The application uses your existing trained models. "
        "No model retraining is performed."
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("---")

st.caption(
    "IMDb Sentiment Analysis | FastText + ANN | Streamlit"
)
