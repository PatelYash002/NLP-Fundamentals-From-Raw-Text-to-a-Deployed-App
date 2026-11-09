# ✈️ Airline Tweet Sentiment Analyzer

### NLP Practical 1 \| Red & White Skill Education

<p align="center">
  <b>From raw airline tweets to sentiment classification using NLP, Machine Learning and Streamlit.</b>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11-blue?logo=python">
  <img alt="NLP" src="https://img.shields.io/badge/Domain-NLP-purple">
  <img alt="Machine Learning" src="https://img.shields.io/badge/ML-Text%20Classification-orange">
  <img alt="Streamlit" src="https://img.shields.io/badge/App-Streamlit-ff4b4b?logo=streamlit">
  <img alt="Status" src="https://img.shields.io/badge/Project-Academic%20Practical-success">
</p>

------------------------------------------------------------------------

## 🌐 Live Demo & Project Links

- 🚀 **Live Streamlit App:** `ADD_YOUR_STREAMLIT_APP_URL`
- 💻 **GitHub Repository:** `ADD_YOUR_GITHUB_REPOSITORY_URL`
- 🎥 **Project Demo Video (5–10 minutes):** `ADD_YOUR_VIDEO_URL`
- 📊 **Notebook:** [`NLP_PR1.ipynb`](./NLP_PR1.ipynb)

> Replace the three `ADD_YOUR_..._URL` placeholders after publishing
> your repository, deploying the app, and uploading your video.

------------------------------------------------------------------------

## 📌 Project Overview

The **Airline Tweet Sentiment Analyzer** is an NLP project that analyzes
real-world airline-related tweets and classifies their sentiment into
three categories:

- 😊 **Positive**
- 😐 **Neutral**
- 😞 **Negative**

The project follows an end-to-end Natural Language Processing workflow:
data exploration, text cleaning, tokenization, sentiment analysis,
feature extraction, Word2Vec embeddings, machine-learning
classification, POS tagging, model evaluation, and a Streamlit
application.

The main goal is to compare different NLP approaches and identify a
model that performs well on unseen airline tweets.

## 🎯 Objectives

- Understand the fundamentals and applications of NLP.
- Explore and clean noisy Twitter text using Regular Expressions.
- Build rule-based and lexicon-based sentiment analysis.
- Apply VADER as an API/library-based sentiment approach.
- Convert text into numerical features using BoW, TF-IDF and N-grams.
- Learn word representations using Word2Vec (CBoW and Skip-gram).
- Train and compare text-classification models.
- Analyze grammatical structure using POS tagging.
- Build a user-facing sentiment prediction app with Streamlit.

## 📂 Dataset

**Dataset:** Twitter US Airline Sentiment  
**Source:** [Kaggle – CrowdFlower Twitter US Airline
Sentiment](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)

The dataset contains approximately **14,640 tweets** and **15 columns**.
This project primarily uses:

| Column              | Description                                                         |
|---------------------|---------------------------------------------------------------------|
| `text`              | Original tweet text; input to the NLP pipeline                      |
| `airline_sentiment` | Target label: negative, neutral or positive                         |
| `airline`           | Airline name; used for exploratory analysis, not as a model feature |

After cleaning and removing unusable/empty cleaned rows, the notebook
reports **14,604 rows**. The train/test split contains **11,683 training
records** and **2,921 test records**, using a stratified 80/20 split
with `random_state=42`.

### ⚖️ Class Distribution

The dataset is imbalanced, with approximately:

- Negative: **62.7%**
- Neutral: **21.2%**
- Positive: **16.1%**

For this reason, the project reports **Macro-F1** in addition to
accuracy so that performance across all three sentiment classes is
considered.

------------------------------------------------------------------------

## 🔄 NLP Workflow

``` text
Raw Airline Tweets
        │
        ▼
Data Loading & Exploration
        │
        ▼
Text Cleaning with RegEx
        │
        ▼
Tokenization & Stop-word Handling
        │
        ▼
Feature Extraction
(BoW • TF-IDF • N-grams • Word2Vec)
        │
        ▼
Sentiment Analysis & Classification
(Heuristic • VADER • Machine Learning)
        │
        ▼
Model Evaluation
(Accuracy • Macro-F1 • Confusion Matrix • Cross-validation)
        │
        ▼
POS Tagging & POS-filtered Features
        │
        ▼
Saved ML Pipeline
        │
        ▼
Streamlit Sentiment Analyzer
```

## 🧠 Tasks & Concepts Covered

| Task | Topic                  | Work performed                                                                   |
|------|------------------------|----------------------------------------------------------------------------------|
| 1    | NLP Fundamentals & EDA | NLP concepts, pipeline, dataset inspection, sentiment and airline analysis       |
| 2    | Text Cleaning          | RegEx cleaning, tokenization, stop-word handling, vocabulary analysis            |
| 3    | NLP Approaches         | Heuristic sentiment, VADER sentiment, test-set evaluation                        |
| 4    | Feature Extraction     | Manual and sklearn BoW, TF-IDF, unigrams, bigrams and trigrams                   |
| 5    | Word Embeddings        | Word2Vec CBoW, Skip-gram, similarity analysis and PCA                            |
| 6    | Text Classification    | BoW/TF-IDF/Word2Vec classifiers, comparison, cross-validation and error analysis |
| 7    | POS Tagging            | POS tags, POS distribution, adjective analysis and POS-filtered model            |
| 8    | Deployment             | Streamlit application and deployment preparation                                 |

------------------------------------------------------------------------

## 📊 Model Performance & Results

All metrics below are taken from the uploaded notebook's test-set
outputs. The split is the same stratified test set of 2,921 records.

### 🏆 Feature and Classifier Comparison

| Features + Classifier                             |   Accuracy |   Macro-F1 |
|---------------------------------------------------|-----------:|-----------:|
| BoW + Naive Bayes                                 |     0.7754 |     0.6958 |
| BoW + Logistic Regression                         |     0.7682 |     0.7204 |
| TF-IDF Unigram + Logistic Regression              |     0.7679 |     0.7211 |
| **TF-IDF Unigram + Bigram + Logistic Regression** | **0.7919** | **0.7402** |
| Word2Vec + Logistic Regression                    |     0.7117 |     0.6588 |

<details>
<summary>📈 Full comparison including Heuristic and VADER</summary>

| Approach                  |   Accuracy |   Macro-F1 |
|---------------------------|-----------:|-----------:|
| TF-IDF + Bigram           | **0.7919** | **0.7402** |
| TF-IDF Unigram            |     0.7679 |     0.7211 |
| BoW + Logistic Regression |     0.7682 |     0.7204 |
| BoW + Naive Bayes         |     0.7754 |     0.6958 |
| Word2Vec                  |     0.7117 |     0.6588 |
| Heuristic                 |     0.4974 |     0.5234 |
| VADER                     |     0.5375 |     0.5078 |

</details>

### ✅ Selected Model

The selected model is **TF-IDF (unigrams + bigrams) + Logistic
Regression**.

- Test accuracy: **79.19%**
- Test Macro-F1: **0.7402**
- 5-fold stratified cross-validation mean Macro-F1: **0.7400**
- Cross-validation standard deviation: **0.0119**
- Scikit-learn version used when saving the model: **1.9.1**

The cross-validation mean and test Macro-F1 are close in this run. The
selected model is saved as
[`sentiment_model.joblib`](./sentiment_model.joblib).

### 🧾 POS-filtered Model Comparison

| Model                                     |     Accuracy |     Macro-F1 |
|-------------------------------------------|-------------:|-------------:|
| Full Text TF-IDF + Logistic Regression    | **0.791852** | **0.740156** |
| POS-filtered TF-IDF + Logistic Regression |     0.705923 |     0.642892 |

**Observation:** POS filtering reduced the Macro-F1 score in this
experiment. Removing nouns and other words can remove useful context for
airline sentiment classification. The full-text model performed better
on this test set.

------------------------------------------------------------------------

## 🖼️ Analysis Outputs & Graphs

All graph images generated in the notebook are included in the
[`outputs/`](./outputs/) folder.

### 1. 📊 Exploratory Data Analysis

**Sentiment Distribution – Counts**

![Sentiment Distribution
Counts](./outputs/sentiment-distribution-counts.png)

**Sentiment Distribution – Percentages**

![Sentiment Distribution
Percentages](./outputs/sentiment-distribution-percentages.png)

**Airline-wise Sentiment Distribution – Stacked Bar Chart**

![Airline-wise
Sentiment](./outputs/airline-wise-sentiment-stacked-bar.png)

**Tweet Length Distribution – Histogram**

![Tweet Length
Distribution](./outputs/tweet-length-distribution-histogram.png)

### 2. 🧹 Text Cleaning & Vocabulary Analysis

**Top 15 Negative and Positive Words**

![Top Words](./outputs/top-15-negative-and-positive-words.png)

**Vocabulary Size vs Maximum Features and N-Grams**

![Vocabulary
Size](./outputs/vocabulary-size-max-features-and-ngrams.png)

**Memory Usage vs Maximum Features**

![Memory Usage](./outputs/memory-usage-vs-maximum-features.png)

### 3. ⚖️ Sentiment Approach & Model Evaluation

**Heuristic and VADER Confusion Matrices**

![Heuristic and VADER Confusion
Matrices](./outputs/heuristic-and-vader-confusion-matrices.png)

**TF-IDF + Logistic Regression Confusion Matrix**

![TF-IDF Confusion
Matrix](./outputs/tfidf-logistic-regression-confusion-matrix.png)

**Macro-F1 Comparison of NLP Approaches**

![Macro-F1
Comparison](./outputs/macro-f1-comparison-of-nlp-approaches.png)

### 4. 🧩 Word2Vec Embeddings

**Word2Vec Embeddings Visualisation Using PCA**

![Word2Vec PCA](./outputs/word2vec-embeddings-pca.png)

### 5. 🏷️ POS Tagging

**POS Distribution by Sentiment**

![POS Distribution](./outputs/pos-distribution-by-sentiment.png)

**Full Text vs POS-Filtered Model Performance**

![POS Filtered
Comparison](./outputs/full-text-vs-pos-filtered-performance.png)

------------------------------------------------------------------------

## 🧰 Technologies & Libraries

| Technology                 | Purpose                                         |
|----------------------------|-------------------------------------------------|
| Python                     | Main programming language                       |
| Pandas & NumPy             | Data handling and numerical operations          |
| Regular Expressions (`re`) | Text cleaning                                   |
| NLTK                       | Tokenization, stop words, VADER and POS tagging |
| Scikit-learn               | Vectorization, classification and evaluation    |
| Gensim                     | Word2Vec CBoW and Skip-gram                     |
| Matplotlib & Seaborn       | Data visualization                              |
| Joblib                     | Saving and loading the trained model            |
| Streamlit                  | Interactive web application                     |
| Jupyter Notebook           | Practical implementation and analysis           |

## 📁 Project Structure

``` text
NLP_PR1/
│
├── NLP_PR1.ipynb                  # Complete Tasks 1–7 notebook
├── app.py                     # Streamlit application
├── nlp_utils.py               # Shared text cleaning and heuristic functions
├── sentiment_model.joblib     # Saved trained ML pipeline
├── Tweets.csv                 # Kaggle dataset used by the notebook
├── requirements.txt           # Python dependencies
│
└── outputs/
    ├── sentiment-distribution-counts.png
    ├── sentiment-distribution-percentages.png
    ├── airline-wise-sentiment-stacked-bar.png
    ├── tweet-length-distribution-histogram.png
    ├── top-15-negative-and-positive-words.png
    ├── vocabulary-size-max-features-and-ngrams.png
    ├── memory-usage-vs-maximum-features.png
    ├── heuristic-and-vader-confusion-matrices.png
    ├── tfidf-logistic-regression-confusion-matrix.png
    ├── macro-f1-comparison-of-nlp-approaches.png
    ├── word2vec-embeddings-pca.png
    ├── pos-distribution-by-sentiment.png
    └── full-text-vs-pos-filtered-performance.png
```

## ⚙️ Installation & Run Locally

### 1. Clone the repository

``` bash
git clone ADD_YOUR_GITHUB_REPOSITORY_URL
cd NLP_PR1
```

### 2. Create and activate a virtual environment (recommended)

``` bash
python -m venv venv
```

Windows:

``` bash
venv\Scripts\activate
```

### 3. Install dependencies

``` bash
pip install -r requirements.txt
```

### 4. Run the notebook

``` bash
jupyter notebook NLP_PR1.ipynb
```

Make sure `Tweets.csv` is in the project root before running the
notebook. The dataset can be downloaded from the Kaggle link above.

### 5. Run the Streamlit application

``` bash
streamlit run app.py
```

The app will open in your browser. Enter an airline-related tweet,
select a sentiment approach, and click **Predict Sentiment**.

## 🌐 Deploy on Streamlit Community Cloud

1.  Push the project to a **public GitHub repository**.
2.  Open [Streamlit Community Cloud](https://share.streamlit.io/).
3.  Sign in with GitHub and select **Create app**.
4.  Choose your repository, branch and `app.py`.
5.  Deploy the application.
6.  Test the live app with positive, negative, neutral and negation
    examples.
7.  Add the public Streamlit URL at the top of this README.

## 🧪 Example Inputs

| Tweet                                                  | Expected sentiment                |
|--------------------------------------------------------|-----------------------------------|
| The flight was amazing and the staff was friendly.     | 😊 Positive                       |
| My flight is delayed and customer service is terrible. | 😞 Negative                       |
| The airline provided information about my booking.     | Model-dependent; check prediction |

The notebook's saved-model test output predicted positive for the first
sentence and negative for the second. For the third sentence, the
notebook predicted negative; this is an example of a model
error/uncertain case, not a guaranteed neutral result.

## ⚠️ Limitations & Future Improvements

- Tweets contain sarcasm, slang, abbreviations and spelling errors that
  are difficult to interpret.
- The dataset has imbalanced sentiment classes.
- Word2Vec mean document vectors do not preserve word order.
- POS taggers trained on news-style text can mislabel informal tweets.
- POS filtering lowered Macro-F1 in this experiment.
- The heuristic approach uses a limited word lexicon and is less capable
  of understanding context and negation than the trained ML pipeline.
- Future work could explore stronger contextual language models, but
  those are outside the implementation scope of this practical.

## 📸 Application Screenshots

Add screenshots of the running Streamlit app to a `screenshots/` folder
before final submission.

Suggested screenshots:

1.  `01-home-screen.png` – Application start screen.
2.  `02-positive-prediction.png` – Positive sentiment output.
3.  `03-negative-prediction.png` – Negative sentiment output.
4.  `04-probability-chart.png` – ML class probability chart.
5.  `05-pos-tags.png` – POS tagging expander, if implemented in the app.

> The uploaded ZIP contains the 13 notebook analysis graph images. It
> does not currently contain these five Streamlit interface screenshots.

## 🎥 Project Demonstration Video

**Video URL:** `ADD_YOUR_VIDEO_URL`

The 5–10 minute video should show your face and screen, explain the main
NLP concepts and measured results, and demonstrate the deployed
Streamlit app with at least three new inputs.

## 👨‍💻 Author

**Yash Patel**  
B.Tech – Computer Engineering  
Bhagwan Mahavir College of Engineering & Technology

**Project:** NLP Practical 1 – Airline Tweet Sentiment Analyzer  
**Institute:** Red & White Skill Education

------------------------------------------------------------------------

<p align="center">
  ✨ <b>Turning Raw Tweets into Meaningful Insights with NLP!</b> ✨
</p>
