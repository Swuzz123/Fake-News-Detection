# Fake sNewss Detection
## Table of Contents
- [Introduction](#introduction)
  - [Data Sources](#data-sources)
  - [Architecture](#architecture)
  - [Prerequisites](#prerequisites)
- [Overview](#overview)
  - [Data Preprocessing](#data-preprocessing)
    - [1. Remove unnecessary columns and Label for the data](#remove-unnecessary-columns-and-label-for-the-data)
    - [2. Check missing and duplicated values](#check-missing-and-duplicated-values)
    - [3. Text cleaning and Normalize text to correct format](#text-cleaning-and-normalize-text-to-correct-format)
  - [Explore Data Analysis (EDA)](#explore-data-analysis-eda)
  - [Model Building and Training](#model-building-and-training)
    - [1. Build with Machine Learning Algorithms](build-with-machine-learning-algorithms)
    - [2. Build with Word2Vec and LSTM](#build-with-word2vec-lstm)
    - [3. Fine tuning with BERT](#fine-tuning-with-bert)
  - [Performance Evaluation](#performance-evaluation)
  - [Model Deployment](#model-deployment)
 
## Introduction
The widespread dissemination of fake news and propaganda presents serious societal risks, including the erosion of public trust, political polarization, manipulation of elections, and the spread of harmful misinformation during crises such as pandemics or conflicts.
From an NLP perspective, detecting fake news is fraught with challenges. Linguistically, fake news often mimics the tone and structure of legitimate journalism, making it difficult to distinguish using surface-level features. 
The absence of reliable and up-to-date labeled datasets, especially across multiple languages and regions, hampers the effectiveness of supervised learning models. 
Additionally, the dynamic and adversarial nature of misinformation means that malicious actors constantly evolve their language and strategies to bypass detection systems. Cultural context, sarcasm, satire, and implicit bias further complicate automated analysis. 
Moreover, NLP models risk amplifying biases present in training data, leading to unfair classifications and potential censorship of legitimate content. 
These challenges underscore the need for cautious, context-aware approaches, as the failure to address them can inadvertently contribute to misinformation, rather than mitigate it.

### Data Sources
- **True Articles:** Reputable media outlets like Reuters, The New York Times, The Washington Post, etc.
- **Fake/Misinformation/Propaganda Articles:**
  - American right-wing extremist websites (e.g., Redflag Newsdesk, Breitbart, Truth Broadcast Network)
  - Public dataset from: Ahmed, H., Traore, I., & Saad, S. (2017): "Detection of Online Fake News Using N-Gram Analysis and Machine Learning Techniques" *(Springer LNCS 10618)*
 
### Architecture
![My image](./images/Architecture_Fake_News_Detection.png)

### Prerequisites
1. Python 3.10.8
  - This setup requires that your machine has python 3.10.8 installed on it. you can refer to this url https://www.python.org/downloads/ to download python. Once you have python downloaded and installed, you will need to setup PATH variables (if you want to run python program directly, detail instructions are below in how to run software section). To do that check this:
  https://www.pythoncentral.io/add-python-to-path-python-is-not-recognized-as-an-internal-or-external-command/.
2. After installing python, you also need to download the required packages
```text
pip install -r requirments.txt
```
