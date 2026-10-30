
import streamlit as st
import joblib
import re
import html
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="IMDB Sentiment Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================
# 2. CUSTOM CSS
# ==========================================

st.markdown("""
<style>

/* Main background */
.stApp {
    background: linear-gradient(
        135deg,
        #0f172a 0%,
        #172554 50%,
        #1e1b4b 100%
    );
    color: white;
}

/* Main container */
.block-container {
    max-width: 1150px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #111827;
    border-right: 1px solid #334155;
}

/* Main heading */
.hero-title {
    font-size: 43px;
    font-weight: 800;
    text-align: center;
    color: #ffffff;
    margin-bottom: 10px;
}

.hero-subtitle {
    text-align: center;
    color: #cbd5e1;
    font-size: 17px;
    margin-bottom: 30px;
}

/* Gradient heading */
.gradient-text {
    background: linear-gradient(
        90deg,
        #a78bfa,
        #38bdf8
    );
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Cards */
.custom-card {
    background: rgba(30, 41, 59, 0.85);
    border: 1px solid #475569;
    border-radius: 18px;
    padding: 22px;
    margin-bottom: 18px;
}

/* Section headings */
.section-heading {
    color: #e2e8f0;
    font-size: 23px;
    font-weight: 700;
    margin-bottom: 15px;
}

/* Text area */
.stTextArea textarea {
    background-color: #1e293b !important;
    color: white !important;
    border: 1px solid #64748b !important;
    border-radius: 12px !important;
    font-size: 16px !important;
}

.stTextArea textarea:focus {
    border: 1px solid #a78bfa !important;
}

/* Buttons */
.stButton > button {
    border-radius: 10px;
    min-height: 43px;
    font-weight: 600;
    border: 1px solid #64748b;
    transition: 0.3s;
}

.stButton > button:hover {
    border-color: #a78bfa;
    color: #c4b5fd;
}

/* Primary button */
.stButton > button[kind="primary"] {
    background: linear-gradient(
        90deg,
        #7c3aed,
        #2563eb
    );
    color: white;
    border: none;
}

/* Result card */
.result-card {
    border-radius: 18px;
    padding: 30px;
    text-align: center;
    margin-top: 20px;
    width: 100%;
    box-sizing: border-box;
}

.positive-card {
    background: linear-gradient(
        135deg,
        #064e3b,
        #065f46
    );
    border: 1px solid #10b981;
}

.negative-card {
    background: linear-gradient(
        135deg,
        #7f1d1d,
        #991b1b
    );
    border: 1px solid #ef4444;
}

.result-icon {
    font-size: 55px;
    margin-bottom: 10px;
}

.result-title {
    font-size: 32px;
    font-weight: 800;
    color: white;
}

.result-description {
    color: #e2e8f0;
    font-size: 16px;
    margin-top: 12px;
}

/* Footer */
.footer {
    text-align: center;
    color: #94a3b8;
    padding: 25px;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. LOAD NLTK RESOURCES
# ==========================================

@st.cache_resource
def load_nltk_resources():

    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

    stop_words = set(stopwords.words('english'))

    # Preserve negative words
    stop_words -= {'no', 'nor', 'not'}

    lemmatizer = WordNetLemmatizer()

    return stop_words, lemmatizer


stop_words, lemmatizer = load_nltk_resources()


# ==========================================
# 4. LOAD TRAINED MODEL
# ==========================================

@st.cache_resource
def load_models():

    model = joblib.load('final_model.pkl')
    tfidf = joblib.load('tfidf_vectorizer.pkl')

    return model, tfidf


model, tfidf = load_models()


# ==========================================
# 5. TEXT PREPROCESSING
# ==========================================

def preprocess_text(text):

    if not text:
        return ""

    text = str(text)

    # Decode HTML entities
    text = html.unescape(text)

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)

    # Remove URLs
    text = re.sub(
        r'https?://\S+|www\.\S+',
        ' ',
        text
    )

    # Lowercase
    text = text.lower()

    # Expand contractions
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

    # Remove special characters
    text = re.sub(r'[^a-z\s]', ' ', text)

    # Tokenization
    tokens = text.split()

    # Remove stopwords
    tokens = [
        word for word in tokens
        if word not in stop_words
    ]

    # Lemmatization
    tokens = [
        lemmatizer.lemmatize(word, pos='v')
        for word in tokens
    ]

    return ' '.join(tokens)


# ==========================================
# 6. SESSION STATE
# ==========================================

if 'review_text' not in st.session_state:
    st.session_state.review_text = ""

if 'result' not in st.session_state:
    st.session_state.result = None

if 'history' not in st.session_state:
    st.session_state.history = []


def set_sample(text):
    st.session_state.review_text = text
    st.session_state.result = None


def clear_review():
    st.session_state.review_text = ""
    st.session_state.result = None


# ==========================================
# 7. SIDEBAR
# ==========================================

with st.sidebar:

    st.markdown("## 🎬 IMDB Analyzer")
    st.markdown("---")

    st.markdown("### 📌 Project Information")

    st.write("**Project:** Movie Review Sentiment Analysis")
    st.write("**Domain:** Machine Learning")
    st.write("**Algorithm:** LinearSVC")
    st.write("**Feature Extraction:** TF-IDF")

    st.markdown("---")

    st.markdown("### ⚙️ Technologies")

    st.write("🐍 Python")
    st.write("🧠 Machine Learning")
    st.write("📊 Scikit-learn")
    st.write("🌐 Streamlit")
    st.write("📝 NLTK")

    st.markdown("---")

    st.markdown("### 📈 Model Performance")

    st.metric(
        label="Test Accuracy",
        value="89.83%"
    )

    st.caption(
        "Previously measured test accuracy. "
        "This is not a live prediction confidence."
    )

    st.markdown("---")

    st.markdown("### 👨‍💻 About")

    st.write(
        "An interactive application that uses "
        "Machine Learning to classify movie "
        "reviews into Positive and Negative sentiments."
    )


# ==========================================
# 8. HERO SECTION
# ==========================================

st.markdown("""
<div class="hero-title">
    🎬 IMDB Movie Review
    <span class="gradient-text">Sentiment Analysis</span>
</div>

<div class="hero-subtitle">
    Discover the emotion behind every movie review.
    <br>
    Powered by Text Preprocessing and Machine Learning.
</div>
""", unsafe_allow_html=True)


# ==========================================
# 9. TOP INFORMATION CARDS
# ==========================================

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="custom-card">
        <h3>🧠 AI Powered</h3>
        <p>Machine Learning based sentiment prediction.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="custom-card">
        <h3>⚡ Fast Analysis</h3>
        <p>Analyze your movie reviews instantly.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="custom-card">
        <h3>🎯 Two Sentiments</h3>
        <p>Classifies reviews as Positive or Negative.</p>
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# 10. SAMPLE REVIEWS
# ==========================================

st.markdown(
    '<div class="section-heading">✨ Try Sample Reviews</div>',
    unsafe_allow_html=True
)

sample_col1, sample_col2 = st.columns(2)

positive_sample = (
    "This movie was absolutely fantastic. "
    "The acting was brilliant, the storyline "
    "was amazing and I really enjoyed it."
)

negative_sample = (
    "This movie was extremely boring. "
    "The acting was terrible, the storyline "
    "was disappointing and I hated it."
)

with sample_col1:
    st.button(
        "😊 Try Positive Review",
        use_container_width=True,
        on_click=set_sample,
        args=(positive_sample,)
    )

with sample_col2:
    st.button(
        "😞 Try Negative Review",
        use_container_width=True,
        on_click=set_sample,
        args=(negative_sample,)
    )


# ==========================================
# 11. REVIEW INPUT
# ==========================================

st.markdown("---")

st.markdown(
    '<div class="section-heading">📝 Write Your Movie Review</div>',
    unsafe_allow_html=True
)

user_review = st.text_area(
    "Enter your review below",
    key="review_text",
    placeholder=(
        "Example: The movie was fantastic! "
        "The actors performed brilliantly..."
    ),
    height=180,
    label_visibility="collapsed"
)

# Character and word count
word_count = len(user_review.split())
character_count = len(user_review)

count_col1, count_col2 = st.columns(2)

with count_col1:
    st.caption(f"📝 Words: {word_count}")

with count_col2:
    st.caption(f"🔠 Characters: {character_count}")


# ==========================================
# 12. ACTION BUTTONS
# ==========================================

button_col1, button_col2 = st.columns([3, 1])

with button_col1:
    predict_button = st.button(
        "🚀 Analyze Sentiment",
        type="primary",
        use_container_width=True
    )

with button_col2:
    st.button(
        "🗑️ Clear",
        use_container_width=True,
        on_click=clear_review
    )


# ==========================================
# 13. PREDICTION
# ==========================================

if predict_button:

    if not user_review.strip():

        st.warning(
            "⚠️ Please enter a movie review first!"
        )

    else:

        with st.spinner("🔍 Analyzing your review..."):

            processed_review = preprocess_text(user_review)

            if not processed_review.strip():

                st.warning(
                    "Please enter a meaningful movie review."
                )

                st.session_state.result = None

            else:

                review_vector = tfidf.transform(
                    [processed_review]
                )

                prediction = model.predict(
                    review_vector
                )[0]

                # Convert model output to sentiment
                if isinstance(prediction, str):

                    sentiment = prediction.strip().lower()

                    if sentiment not in ['positive', 'negative']:
                        st.error("Unexpected model output.")
                        st.stop()

                else:

                    sentiment = (
                        "positive" if int(prediction) == 1
                        else "negative"
                    )

                st.session_state.result = {
                    "sentiment": sentiment,
                    "processed_review": processed_review,
                    "original_review": user_review
                }

                st.session_state.history.insert(
                    0,
                    {
                        "review": user_review[:100],
                        "sentiment": sentiment.title()
                    }
                )

                st.session_state.history = (
                    st.session_state.history[:10]
                )


# ==========================================
# 14. DISPLAY PREDICTION RESULT
# ==========================================

if st.session_state.result is not None:

    result = st.session_state.result
    sentiment = result["sentiment"]

    st.markdown("---")

    st.markdown(
        '<div class="section-heading">📊 Sentiment Analysis Result</div>',
        unsafe_allow_html=True
    )

    if sentiment == "positive":

        positive_html = """
        <div class="result-card positive-card">
            <div class="result-icon">😊</div>
            <div class="result-title">POSITIVE REVIEW</div>
            <p class="result-description">
                Great! The model identifies this review
                as expressing a positive sentiment.
            </p>
        </div>
        """

        st.html(positive_html)
        st.balloons()

    else:

        negative_html = """
        <div class="result-card negative-card">
            <div class="result-icon">😞</div>
            <div class="result-title">NEGATIVE REVIEW</div>
            <p class="result-description">
                The model identifies this review
                as expressing a negative sentiment.
            </p>
        </div>
        """

        st.html(negative_html)


# ==========================================
# 15. PREDICTION HISTORY
# ==========================================

if st.session_state.history:

    st.markdown("---")

    st.markdown(
        '<div class="section-heading">📚 Recent Prediction History</div>',
        unsafe_allow_html=True
    )

    for index, item in enumerate(st.session_state.history):

        col1, col2 = st.columns([4, 1])

        with col1:
            st.write(
                f"**{index + 1}.** {item['review']}"
            )

        with col2:

            if item["sentiment"] == "Positive":
                st.success("😊 Positive")
            else:
                st.error("😞 Negative")


# ==========================================
# 16. FOOTER
# ==========================================

st.markdown("---")

st.markdown("""
<div class="footer">
    <h3>🎬 IMDB Movie Review Sentiment Analysis</h3>
    Developed using Python | Text Preprocessing | TF-IDF | LinearSVC | Streamlit
    <br><br>
    Machine Learning Project
</div>
""", unsafe_allow_html=True)
