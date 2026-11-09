
import streamlit as st
import joblib
import nltk
import pandas as pd
import re
from pathlib import Path

from nlp_utils import clean_text, heuristic_sentiment
from nltk.sentiment import SentimentIntensityAnalyzer


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Airline Sentiment Analyzer",
    page_icon="✈️",
    layout="centered"
)


# ==========================================
# DOWNLOAD REQUIRED NLTK RESOURCES
# ==========================================

for package in [
    "vader_lexicon",
    "punkt",
    "averaged_perceptron_tagger"
]:
    try:
        nltk.download(package, quiet=True)
    except Exception:
        pass


# ==========================================
# PAGE TITLE
# ==========================================

st.title("✈️ Airline Tweet Sentiment Analyzer")

st.write(
    "Analyze the sentiment of airline-related tweets "
    "using NLP and Machine Learning."
)

st.divider()


# ==========================================
# LOAD ML MODEL
# ==========================================

@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "sentiment_model.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            "sentiment_model.joblib not found. "
            "Please place it in the project folder."
        )

    return joblib.load(model_path)


# ==========================================
# MODEL COMPATIBILITY FIX
# ==========================================

def fix_model_compatibility(model):

    # Handle sklearn Pipeline
    if hasattr(model, "steps"):
        classifier = model.steps[-1][1]
    else:
        classifier = model

    # Fix missing multi_class attribute
    if classifier.__class__.__name__ == "LogisticRegression":

        if not hasattr(classifier, "multi_class"):
            classifier.multi_class = "auto"

    return model


# ==========================================
# EXAMPLE TWEETS
# ==========================================

examples = [
    "The flight was amazing and the staff was helpful!",
    "My flight is delayed and the service is terrible.",
    "The flight was okay, nothing special."
]


st.subheader("Try an Example")

for i, example in enumerate(examples):

    if st.button(f"Example {i + 1}", key=f"example_{i}"):
        st.session_state["tweet"] = example


# ==========================================
# USER INPUT
# ==========================================

tweet = st.text_area(
    "Enter your tweet:",
    value=st.session_state.get("tweet", ""),
    height=120,
    max_chars=1000,
    placeholder="Example: My flight experience was excellent!"
)


# ==========================================
# SELECT SENTIMENT APPROACH
# ==========================================

approach = st.selectbox(
    "Select Sentiment Approach",
    ["ML Model", "VADER", "Heuristic"]
)


# ==========================================
# SENTIMENT PREDICTION
# ==========================================

if st.button("Predict Sentiment"):

    if not tweet.strip():

        st.warning("Please enter a tweet first.")

    else:

        cleaned = clean_text(tweet)

        if not cleaned:

            st.warning(
                "No meaningful text found. Please enter a valid tweet."
            )

        else:

            try:

                # ----------------------------------
                # HEURISTIC APPROACH
                # ----------------------------------

                if approach == "Heuristic":

                    result = heuristic_sentiment(tweet)

                    st.write("Approach: Heuristic")


                # ----------------------------------
                # VADER APPROACH
                # ----------------------------------

                elif approach == "VADER":

                    analyzer = SentimentIntensityAnalyzer()

                    score = analyzer.polarity_scores(tweet)["compound"]

                    if score >= 0.05:
                        result = "positive"

                    elif score <= -0.05:
                        result = "negative"

                    else:
                        result = "neutral"

                    st.write("Approach: VADER")

                    st.subheader("VADER Sentiment Score")

                    scores = analyzer.polarity_scores(tweet)

                    score_df = pd.DataFrame({
                        "Score": scores
                    })

                    st.bar_chart(score_df)


                # ----------------------------------
                # MACHINE LEARNING APPROACH
                # ----------------------------------

                
                else:
                    model = load_model()

                    # Fix LogisticRegression compatibility
                    if hasattr(model, "steps"):
                        classifier = model.steps[-1][1]
                    else:
                        classifier = model

                    # Explicitly restore missing attribute
                    if classifier.__class__.__name__ == "LogisticRegression":
                        classifier.multi_class = "auto"

                    # Predict sentiment
                    result = str(model.predict([cleaned])[0]).lower()

                    # Prediction probabilities
                    if hasattr(model, "predict_proba"):

                        probabilities = model.predict_proba([cleaned])[0]

                        probability_df = pd.DataFrame({
                            "Sentiment": model.classes_,
                            "Probability": probabilities
                        })

                        st.subheader("Prediction Probabilities")

                        st.bar_chart(
                            probability_df.set_index("Sentiment")
                        )

                    st.write("Approach: ML Model")



                # ----------------------------------
                # DISPLAY RESULT
                # ----------------------------------

                st.divider()

                st.subheader("Sentiment Analysis Result")

                if result == "positive":

                    st.success("😊 Positive Sentiment")

                elif result == "negative":

                    st.error("😞 Negative Sentiment")

                else:

                    st.info("😐 Neutral Sentiment")


                # ----------------------------------
                # DISPLAY ORIGINAL TWEET
                # ----------------------------------

                st.subheader("Analyzed Tweet")

                st.write(tweet)

                st.caption(
                    f"Selected Approach: {approach}"
                )


            except FileNotFoundError as e:

                st.error(str(e))

            except Exception as e:

                st.error(f"Prediction Error: {e}")

                st.info(
                    "Please check your trained model and "
                    "Scikit-learn version."
                )


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "✈️ Airline Tweet Sentiment Analyzer | "
    "NLP & Machine Learning Project"
)
