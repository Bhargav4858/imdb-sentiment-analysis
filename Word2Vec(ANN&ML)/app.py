import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re
import html
import nltk
import joblib
import textwrap

from pathlib import Path
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="IMDb Sentiment AI Dashboard",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent

WORD2VEC_PATH = BASE_DIR / "imdb_word2vec.model"
ANN_PATH = BASE_DIR / "IMDB_Sentiment_Analysis_(Word2Vec&ANN).keras"
ENCODER_PATH = BASE_DIR / "imdb_label_encoder.pkl"


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

/* Main application background */

.stApp {
    background-color: #F4F7FB;
}

/* Normal text */

.stApp p,
.stApp li,
.stApp label {
    color: #263445;
}

/* Headings */

.stApp h1,
.stApp h2,
.stApp h3,
.stApp h4,
.stApp h5,
.stApp h6 {
    color: #172B4D !important;
    font-weight: 700 !important;
}

/* Sidebar */

section[data-testid="stSidebar"] {
    background-color: #EAF0F8;
}

section[data-testid="stSidebar"] * {
    color: #263445;
}

/* Hero section */

.hero {
    padding: 30px;
    border-radius: 16px;
    background: linear-gradient(120deg, #14213D, #315A91);
    margin-bottom: 25px;
}

.hero h1 {
    color: #FFFFFF !important;
    font-size: 32px !important;
    font-weight: 700 !important;
    margin-bottom: 12px;
}

.hero p {
    color: #E5EDFA !important;
    font-size: 16px;
    margin-bottom: 0;
}

/* Metric cards */

div[data-testid="stMetric"] {
    background-color: #FFFFFF !important;
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #DCE4EF;
    box-shadow: 0 3px 10px rgba(0,0,0,0.04);
}

div[data-testid="stMetric"] label,
div[data-testid="stMetric"] label *,
div[data-testid="stMetricValue"],
div[data-testid="stMetricValue"] *,
div[data-testid="stMetricDelta"],
div[data-testid="stMetricDelta"] * {
    color: #172B4D !important;
}

/* Section headings */

.section-title {
    color: #172B4D !important;
    font-size: 21px;
    font-weight: 700;
}

/* Data tables */

div[data-testid="stDataFrame"] {
    background-color: #FFFFFF;
    border-radius: 10px;
}

[data-testid="stTable"] th {
    background-color: #E5EDF7 !important;
    color: #172B4D !important;
}

[data-testid="stTable"] td {
    background-color: #FFFFFF !important;
    color: #263445 !important;
}

/* Input fields */

.stTextArea textarea,
.stTextInput input {
    background-color: #FFFFFF !important;
    color: #172B4D !important;
    border: 1px solid #C8D3E0 !important;
}

.stTextArea textarea::placeholder {
    color: #718096 !important;
}

/* Select boxes */

div[data-baseweb="select"] * {
    color: #263445 !important;
}

/* Positive result */

.result-positive {
    padding: 20px;
    border-radius: 12px;
    background-color: #E4F7EC;
    border: 1px solid #A8DFBD;
    color: #126B37 !important;
    font-size: 22px;
    font-weight: bold;
}

/* Negative result */

.result-negative {
    padding: 20px;
    border-radius: 12px;
    background-color: #FDECEC;
    border: 1px solid #F2B6B6;
    color: #A32626 !important;
    font-size: 22px;
    font-weight: bold;
}

/* Buttons */

.stButton button,
.stFormSubmitButton button {
    background-color: #2457A7;
    color: #FFFFFF !important;
    border-radius: 9px;
    border: none;
    font-weight: 600;
}

.stButton button:hover,
.stFormSubmitButton button:hover {
    background-color: #173F80;
    color: #FFFFFF !important;
}

/* Download button */

.stDownloadButton button {
    background-color: #16845B;
    color: #FFFFFF !important;
}

/* Captions */

[data-testid="stCaptionContainer"] {
    color: #64748B !important;
}

/* Code blocks */

[data-testid="stCodeBlock"] pre,
[data-testid="stCodeBlock"] code {
    background-color: #FFFFFF !important;
    color: #263445 !important;
}

/* Divider */

hr {
    border-color: #D8E1EC;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# NLTK RESOURCES
# =========================================================

@st.cache_resource
def load_nltk_resources():

    resources = [
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4")
    ]

    for resource, package in resources:

        try:
            nltk.data.find(resource)

        except LookupError:
            nltk.download(package, quiet=True)

    words = set(stopwords.words("english"))

    # Preserve negation words
    words -= {"no", "nor", "not"}

    return words, WordNetLemmatizer()


stop_words, lemmatizer = load_nltk_resources()


# =========================================================
# LOAD TRAINED MODELS
# =========================================================

@st.cache_resource
def load_artifacts():

    required_files = [
        WORD2VEC_PATH,
        ANN_PATH,
        ENCODER_PATH
    ]

    missing_files = [
        file_path.name
        for file_path in required_files
        if not file_path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Missing model files: " + ", ".join(missing_files)
        )

    word2vec_model = Word2Vec.load(
        str(WORD2VEC_PATH)
    )

    ann_model = load_model(
        str(ANN_PATH),
        compile=False
    )

    label_encoder = joblib.load(
        ENCODER_PATH
    )

    return word2vec_model, ann_model, label_encoder


try:

    word2vec_model, ann_model, label_encoder = load_artifacts()

except Exception as error:

    st.error(f"Model loading failed: {error}")

    st.warning(
        "Check that all model files are in the same folder as app.py "
        "and that your Python package versions are compatible."
    )

    st.stop()


VECTOR_SIZE = word2vec_model.wv.vector_size


# =========================================================
# TEXT PREPROCESSING
# =========================================================

def preprocess_text(review):

    if pd.isna(review):
        return []

    text = str(review)

    text = html.unescape(text)

    text = re.sub(r"<[^>]+>", " ", text)

    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    text = text.lower()

    # Contraction handling

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

    text = re.sub(r"[^a-z\s]", " ", text)

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


# =========================================================
# WORD2VEC VECTORIZATION
# =========================================================

def review_to_vector(tokens):

    vectors = [
        word2vec_model.wv[word]
        for word in tokens
        if word in word2vec_model.wv
    ]

    if vectors:

        return np.mean(
            vectors,
            axis=0
        ).astype(np.float32)

    return np.zeros(
        VECTOR_SIZE,
        dtype=np.float32
    )


# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict_sentiment(review):

    tokens = preprocess_text(review)

    vector = review_to_vector(tokens)

    input_vector = np.asarray(
        [vector],
        dtype=np.float32
    )

    prediction = ann_model.predict(
        input_vector,
        verbose=0
    )

    probability = float(
        np.asarray(prediction).reshape(-1)[0]
    )

    predicted_index = 1 if probability >= 0.5 else 0

    sentiment = str(
        label_encoder.inverse_transform(
            [predicted_index]
        )[0]
    )

    confidence = (
        probability
        if predicted_index == 1
        else 1 - probability
    )

    return {
        "sentiment": sentiment,
        "confidence": confidence,
        "probability": probability,
        "tokens": len(tokens)
    }


# =========================================================
# SESSION STATE
# =========================================================

if "history" not in st.session_state:

    st.session_state.history = []


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🎬 IMDb Sentiment AI")

st.sidebar.caption(
    "Word2Vec + Artificial Neural Network"
)

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Dashboard",
        "Sentiment Prediction",
        "Batch Analysis",
        "Prediction History",
        "Model Insights"
    ]
)

st.sidebar.divider()

st.sidebar.success("Word2Vec Model Loaded")

st.sidebar.success("ANN Model Loaded")

st.sidebar.caption(
    f"Embedding Dimension: {VECTOR_SIZE}"
)

if st.sidebar.button("Clear Prediction History"):

    st.session_state.history = []

    st.rerun()


# =========================================================
# HERO SECTION
# =========================================================

hero_html = textwrap.dedent("""
<div class="hero">
    <h1>🎬 IMDb Sentiment Analysis</h1>
    <p>
        Understand movie reviews using Word2Vec embeddings
        and Artificial Neural Networks.
    </p>
</div>
""").strip()

st.markdown(
    hero_html,
    unsafe_allow_html=True
)


# =========================================================
# DASHBOARD
# =========================================================

if page == "Dashboard":

    st.header("📊 Dashboard Overview")

    history = pd.DataFrame(
        st.session_state.history
    )

    total = len(history)

    if total > 0:

        positive = (
            history["Sentiment"].str.lower() == "positive"
        ).sum()

        negative = total - positive

        average_confidence = history["Confidence"].mean()

    else:

        positive = 0
        negative = 0
        average_confidence = 0.0

    # KPI cards

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Predictions",
        total
    )

    c2.metric(
        "Positive Reviews",
        int(positive)
    )

    c3.metric(
        "Negative Reviews",
        int(negative)
    )

    c4.metric(
        "Average Confidence",
        f"{average_confidence:.1f}%"
    )

    st.divider()

    left, right = st.columns(2)

    # Sentiment distribution

    with left:

        st.subheader("Sentiment Distribution")

        if total > 0:

            sentiment_data = pd.DataFrame({
                "Sentiment": ["Positive", "Negative"],
                "Count": [positive, negative]
            })

            fig = px.pie(
                sentiment_data,
                names="Sentiment",
                values="Count",
                hole=0.55,
                title="Positive vs Negative Reviews",
                color="Sentiment",
                color_discrete_map={
                    "Positive": "#20A36B",
                    "Negative": "#E05252"
                }
            )

            fig.update_traces(
                textinfo="label+percent"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            st.info(
                "Make your first prediction to see sentiment distribution."
            )

    # Confidence distribution

    with right:

        st.subheader("Prediction Confidence")

        if total > 0:

            fig = px.histogram(
                history,
                x="Confidence",
                nbins=10,
                title="Confidence Score Distribution",
                labels={
                    "Confidence": "Confidence (%)"
                }
            )

            fig.update_layout(
                xaxis_title="Confidence (%)",
                yaxis_title="Number of Reviews"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            st.info(
                "Confidence analytics will appear after predictions."
            )

    st.divider()

    st.subheader("Recent Predictions")

    if total > 0:

        st.dataframe(
            history.tail(10).iloc[::-1],
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info("No predictions available yet.")


# =========================================================
# SINGLE REVIEW PREDICTION
# =========================================================

elif page == "Sentiment Prediction":

    st.header("🔍 Analyze a Movie Review")

    st.write(
        "Enter a movie review below and let the ANN model "
        "classify its sentiment."
    )

    with st.form("prediction_form"):

        review = st.text_area(
            "Movie Review",
            placeholder=(
                "Example: The movie was amazing. "
                "The acting and storyline were excellent!"
            ),
            height=180
        )

        submitted = st.form_submit_button(
            "🚀 Analyze Sentiment",
            type="primary",
            use_container_width=True
        )

    if submitted:

        if not review.strip():

            st.warning("Please enter a review first.")

        else:

            with st.spinner("Analyzing your review..."):

                result = predict_sentiment(review)

            sentiment = result["sentiment"]

            confidence = result["confidence"]

            st.divider()

            st.subheader("Prediction Result")

            if sentiment.lower() == "positive":

                st.markdown(
                    '<div class="result-positive">'
                    '😊 POSITIVE REVIEW'
                    '</div>',
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    '<div class="result-negative">'
                    '😞 NEGATIVE REVIEW'
                    '</div>',
                    unsafe_allow_html=True
                )

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Sentiment",
                sentiment.upper()
            )

            c2.metric(
                "Confidence",
                f"{confidence * 100:.2f}%"
            )

            c3.metric(
                "Processed Tokens",
                result["tokens"]
            )

            st.write("Prediction Confidence")

            st.progress(
                float(confidence)
            )

            st.caption(
                "Confidence is the model's predicted probability "
                "for the selected class, not a guarantee of correctness."
            )

            st.session_state.history.append({
                "Review": review,
                "Sentiment": sentiment,
                "Confidence": round(confidence * 100, 2)
            })

            with st.expander("View preprocessing details"):

                tokens = preprocess_text(review)

                st.write("Processed Tokens:")

                st.write(tokens)

                st.write(
                    f"Word2Vec Vector Dimension: {VECTOR_SIZE}"
                )

                st.write(
                    f"Raw Sigmoid Probability: "
                    f"{result['probability']:.4f}"
                )


# =========================================================
# BATCH ANALYSIS
# =========================================================

elif page == "Batch Analysis":

    st.header("📁 Batch Sentiment Analysis")

    st.write(
        "Upload a CSV file with a column named `review`."
    )

    uploaded_file = st.file_uploader(
        "Upload your CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            df = pd.read_csv(uploaded_file)

            if "review" not in df.columns:

                st.error(
                    "Please include a column named 'review'."
                )

            elif df.empty:

                st.warning("The uploaded CSV is empty.")

            else:

                st.subheader("Uploaded Reviews")

                st.dataframe(
                    df.head(10),
                    use_container_width=True
                )

                st.metric(
                    "Number of Reviews",
                    len(df)
                )

                if st.button(
                    "Analyze All Reviews",
                    type="primary",
                    use_container_width=True
                ):

                    sentiments = []

                    confidences = []

                    progress = st.progress(0)

                    status = st.empty()

                    for i, review_item in enumerate(df["review"]):

                        result = predict_sentiment(review_item)

                        sentiments.append(
                            result["sentiment"]
                        )

                        confidences.append(
                            round(
                                result["confidence"] * 100,
                                2
                            )
                        )

                        progress.progress(
                            (i + 1) / len(df)
                        )

                        status.text(
                            f"Processing review {i + 1} "
                            f"of {len(df)}"
                        )

                    result_df = df.copy()

                    result_df["Predicted Sentiment"] = sentiments

                    result_df["Confidence (%)"] = confidences

                    # Add results to session history

                    for _, row in result_df.iterrows():

                        st.session_state.history.append({

                            "Review": str(row["review"]),

                            "Sentiment": row["Predicted Sentiment"],

                            "Confidence": row["Confidence (%)"]

                        })

                    status.success("Analysis completed!")

                    st.subheader("Analysis Results")

                    st.dataframe(
                        result_df,
                        use_container_width=True
                    )

                    positive_count = sum(
                        str(value).lower() == "positive"
                        for value in sentiments
                    )

                    negative_count = (
                        len(sentiments) - positive_count
                    )

                    c1, c2 = st.columns(2)

                    c1.metric(
                        "Positive Reviews",
                        positive_count
                    )

                    c2.metric(
                        "Negative Reviews",
                        negative_count
                    )

                    # Batch chart

                    chart_data = pd.DataFrame({

                        "Sentiment": [
                            "Positive",
                            "Negative"
                        ],

                        "Count": [
                            positive_count,
                            negative_count
                        ]

                    })

                    fig = px.bar(
                        chart_data,
                        x="Sentiment",
                        y="Count",
                        color="Sentiment",
                        title="Batch Sentiment Distribution",
                        text="Count",
                        color_discrete_map={
                            "Positive": "#20A36B",
                            "Negative": "#E05252"
                        }
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

                    csv_data = result_df.to_csv(
                        index=False
                    ).encode("utf-8")

                    st.download_button(
                        "⬇️ Download Results",
                        data=csv_data,
                        file_name="word2vec_sentiment_results.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

        except Exception as error:

            st.error(
                f"Error processing the uploaded file: {error}"
            )


# =========================================================
# PREDICTION HISTORY
# =========================================================

elif page == "Prediction History":

    st.header("📜 Prediction History")

    history = pd.DataFrame(
        st.session_state.history
    )

    if not history.empty:

        st.metric(
            "Total Reviews Analyzed",
            len(history)
        )

        st.dataframe(
            history.iloc[::-1],
            use_container_width=True,
            hide_index=True
        )

        csv_data = history.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "Download Prediction History",
            data=csv_data,
            file_name="prediction_history.csv",
            mime="text/csv"
        )

    else:

        st.info(
            "Your prediction history will appear here "
            "after analyzing reviews."
        )


# =========================================================
# MODEL INSIGHTS
# =========================================================

elif page == "Model Insights":

    st.header("🧠 Model Insights")

    # Word2Vec Configuration

    st.subheader("Word2Vec Configuration")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Vector Size",
        VECTOR_SIZE
    )

    c2.metric(
        "Vocabulary Size",
        f"{len(word2vec_model.wv):,}"
    )

    c3.metric(
        "Training Algorithm",
        "Skip-gram" if word2vec_model.sg == 1 else "CBOW"
    )

    st.divider()

    # ANN Architecture

    st.subheader("ANN Architecture")

    architecture = f"""
Input Layer ({VECTOR_SIZE} features)
          |
Dense Layer (32 neurons, ReLU)
          |
Dense Layer (16 neurons, ReLU)
          |
Output Layer (1 neuron, Sigmoid)
"""

    st.code(
        architecture,
        language="text"
    )

    st.divider()

    # Model Information

    st.subheader("Model Information")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("**ANN Input Shape**")

        st.code(
            str(ann_model.input_shape),
            language="text"
        )

        st.markdown("**ANN Output Shape**")

        st.code(
            str(ann_model.output_shape),
            language="text"
        )

    with col2:

        st.markdown("**Word2Vec Vocabulary**")

        st.code(
            f"{len(word2vec_model.wv):,}",
            language="text"
        )

        st.markdown("**Label Encoder Classes**")

        st.code(
            str(list(label_encoder.classes_)),
            language="text"
        )

    st.divider()

    # Training Configuration

    st.subheader("Training Configuration")

    training_info = textwrap.dedent("""
    - Optimizer: Adam
    - Loss Function: Binary Crossentropy
    - Hidden Layers: 32 and 16 neurons
    - Hidden Activation: ReLU
    - Output Activation: Sigmoid
    """).strip()

    st.markdown(training_info)


# =========================================================
# FOOTER
# =========================================================

st.divider()

footer_html = textwrap.dedent("""
<div style="text-align:center; padding: 8px 0;">
    <p style="color:#64748B;">
        IMDb Movie Review Sentiment Analysis |
        Word2Vec + ANN |
        Built with Python and Streamlit
    </p>
</div>
""").strip()

st.markdown(
    footer_html,
    unsafe_allow_html=True
)