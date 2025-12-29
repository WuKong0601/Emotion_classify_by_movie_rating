# A Study of Vietnamese Sentiment Classification with Ensemble Pre-Trained Language Models

**Dang Van Thin\***, **Duong Ngoc Hao†**, **Ngan Luu-Thuy Nguyen‡**  
University of Information Technology  
Vietnam National University Ho Chi Minh City  

Emails:  
- thindv@uit.edu.vn  
- haodn@uit.edu.vn  
- ngannlt@uit.edu.vn  

\* Corresponding author

---

## Abstract

Sentiment Analysis (SA) has attracted increasing research attention in recent years. Most existing works tackle the SA task by fine-tuning single pre-trained language models combined with specific layers. Despite their effectiveness, previous studies overlooked the utilization of feature representations from various contextual language models.

This paper introduces **two ensemble frameworks**—**feature fusion** and **soft voting**—that combine multiple transformer-based language models, including **PhoBERT, XLM-R, XLM-Align, InfoXLM, and viBERT FPT**. Extensive experiments on Vietnamese benchmarks at **sentence-level, document-level, and aspect-level SA** demonstrate that the proposed approaches outperform existing methods and achieve new state-of-the-art results.

**Keywords:** Sentiment analysis; aspect-based sentiment analysis; ensemble learning; Vietnamese NLP

---

## 1. Introduction

With the rapid growth of online platforms, a massive amount of user-generated reviews and opinions has become available. Extracting sentiment information from such data is crucial for decision-making in business and society.

Sentiment Analysis (SA) is studied at:
- **Coarse-grained level:** document-level, sentence-level
- **Fine-grained level:** Aspect-based Sentiment Analysis (ABSA)

While SA has been widely explored for English and Chinese, Vietnamese SA remains challenging due to:
- Limited high-quality datasets
- Complex linguistic structure (multi-syllable words)
- Scarcity of strong ensemble approaches for ABSA

Recent transformer-based language models (e.g., BERT, XLM-R, PhoBERT) provide contextual representations that vary by training data and objectives. Leveraging **heterogeneous ensembles** of such models can enhance performance.

### Contributions

- Propose **two ensemble frameworks**: feature fusion & soft voting
- Combine **monolingual and multilingual PLMs**
- Evaluate on **5 Vietnamese benchmark datasets**
- Achieve **state-of-the-art results** on 4/5 datasets

---

## 2. Related Work

### 2.1 Sentiment Analysis

Early Vietnamese SA relied on:
- Sentiment dictionaries
- Ontology-based methods
- Traditional ML models (SVM, MaxEnt)

Later approaches employed:
- CNN, LSTM, BiLSTM-CNN
- Transfer learning with BERT-based models
- Limited ensemble methods

Recent studies show **PhoBERT** outperforms multilingual models for Vietnamese SA.

---

### 2.2 Aspect-Based Sentiment Analysis (ABSA)

ABSA research includes:
- Aspect category detection
- Aspect sentiment classification
- Joint and multi-task learning approaches

Vietnamese ABSA has adopted:
- BiLSTM + CRF
- CNN-based multi-label classification
- Fine-tuning PhoBERT and mBERT

However, ensemble methods using multiple PLMs remain underexplored.

---

### 2.3 Ensemble Techniques in NLP

Ensemble methods:
- Voting
- Stacking
- Bagging
- Feature fusion

Prior Vietnamese works mainly combine:
- Deep learning models
- Limited transformer ensembles

This paper fills the gap by **ensembling heterogeneous transformer-based PLMs**.

---

## 3. Methodology

### 3.1 Problem Definition

#### Sentiment Analysis (SA)

Given a review:

