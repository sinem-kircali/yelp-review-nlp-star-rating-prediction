# Yelp Review NLP: Word Embeddings, Sentiment & Star Rating Prediction

NLP pipeline applied to Yelp restaurant reviews — built as a graded assignment for the *Large Language Models for Marketing* course at Erasmus University Rotterdam.

## Pipeline
1. **Data Preparation** — stratified sample of 6,000 reviews (1,200 per star level, 1–5) from a 19,896-review Yelp dataset; ensures balanced representation across satisfaction levels
2. **Word Embeddings (Word2Vec)** — train custom embeddings on review text; interpret semantic directions and dimensions
3. **Topic Modelling** — BERTopic to extract latent topics across reviews
4. **Sentiment Analysis** — VADER sentence-level sentiment scoring
5. **Regression** — OLS models linking embedding features, topic presence, and sentiment scores to star ratings

## Tech Stack
- Python 3
- `gensim` (Word2Vec) · `bertopic` · `vaderSentiment`
- `statsmodels` · `scikit-learn`
- `pandas` · `matplotlib` · `seaborn`

## Data
Yelp Restaurant Reviews (~20K reviews, publicly available on Kaggle). Stratified sampling ensures equal representation of each star rating class.
