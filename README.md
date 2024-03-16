
# Disaster Tweet Classifier with Uncertainty

This project explores incorporating uncertainty into binary classification of tweets, specifically identifying tweets related to disasters. It experiments with blending traditional NLP techniques and Bayesian inference to provide not only a classification but also a measure of confidence in that classification.

## Architecture Overview

- **NLP Model:** Utilizes GPT-2 for converting tweets into embeddings, serving as a nuanced representation of the textual data.
- **Classification Model:** A neural network built with PyTorch that takes GPT-2 embeddings as input and predicts two parameters (alpha and beta) of a Beta distribution. This distribution represents the probability of a tweet being disaster-related along with the uncertainty of the prediction.
    - **Input Layer:** Receives the GPT-2 embeddings.
    - **Hidden Layer(s):** Processes the embeddings through one or more layers with non-linear activation functions.
    - **Output Layer:** Outputs two values, alpha and beta, parameters of the Beta distribution.
- **Loss Function:** A custom loss function that combines Binary Cross-Entropy (BCE) for accuracy and a penalty for variance, encouraging the model to be both accurate and confident in its predictions.

## Quick Start

To get started, clone the repository, set up a virtual environment, install dependencies, and run the provided Jupyter notebook:

```bash
git clone <repository-url>
cd <repository-directory>
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
jupyter notebook disaster_tweet_classifier.ipynb
```