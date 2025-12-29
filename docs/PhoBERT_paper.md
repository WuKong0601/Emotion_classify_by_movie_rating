# PhoBERT: Pre-trained Language Models for Vietnamese

**Dat Quoc Nguyen¹**, **Anh Tuan Nguyen²***  
¹ VinAI Research, Vietnam  
² NVIDIA, USA  

Emails:  
- v.datnq9@vinai.io  
- tuananhn@nvidia.com  

\* Work done during internship at VinAI Research.

---

## Abstract

We present **PhoBERT** with two versions—**PhoBERTbase** and **PhoBERTlarge**—the first public large-scale monolingual language models pre-trained for Vietnamese. Experimental results show that PhoBERT consistently outperforms the recent best pre-trained multilingual model **XLM-R** and improves the state-of-the-art in multiple Vietnamese-specific NLP tasks including Part-of-speech tagging, Dependency parsing, Named-entity recognition and Natural language inference.

We publicly release PhoBERT to facilitate future research and downstream applications for Vietnamese NLP.

Repository: https://github.com/VinAIResearch/PhoBERT

---

## 1. Introduction

Pre-trained language models, especially **BERT** (Devlin et al., 2019), have recently led to significant performance improvements across NLP tasks. However, such improvements are mostly limited to English.

For Vietnamese language modeling, there are two major concerns:

1. **Limited pre-training data**  
   Vietnamese Wikipedia is the primary corpus used for training monolingual and multilingual models, but it is small (~1GB) and not representative of general language use.

2. **Syllable vs. word ambiguity**  
   Vietnamese whitespace separates syllables, not words. Existing models apply BPE directly on syllable-level data, which may degrade performance on word-level tasks.

To address these issues, we train **PhoBERT**, the first large-scale monolingual Vietnamese language models, using **20GB of word-level Vietnamese data**.

### Contributions

- The first large-scale monolingual BERT-based models for Vietnamese
- State-of-the-art performance on POS tagging, Dependency parsing, NER, and NLI
- Comprehensive comparison between monolingual PhoBERT and multilingual XLM-R
- Public release of PhoBERT models compatible with `fairseq` and `transformers`

---

## 2. PhoBERT

### Architecture

PhoBERT follows the **RoBERTa** pre-training approach and has two versions:

- **PhoBERTbase** (same architecture as BERTbase)
- **PhoBERTlarge** (same architecture as BERTlarge)

---

### Pre-training Data

To address data scarcity, we construct a **20GB Vietnamese corpus**, consisting of:

1. Vietnamese Wikipedia (~1GB)
2. Vietnamese news corpus (~19GB), deduplicated from an original 50GB dataset

To address syllable ambiguity, we apply **RDRSegmenter** from **VnCoreNLP** to perform word segmentation before BPE.

Statistics:
- ~145M word-segmented sentences
- ~3B word tokens
- Vocabulary size: 64K subword units
- Average: 24.4 subword tokens per sentence

---

### Optimization

- Framework: `fairseq`
- Token length: max 256 subwords
- Optimizer: Adam
- Epochs: 40
- Hardware: 4× V100 GPUs (16GB)

| Model | Batch Size | Peak LR | Steps | Training Time |
|-----|-----------|--------|------|---------------|
| PhoBERTbase | 1024 | 0.0004 | ~540K | ~3 weeks |
| PhoBERTlarge | 512 | 0.0002 | ~1.08M | ~5 weeks |

---

## 3. Experimental Setup

PhoBERT is evaluated on four Vietnamese NLP tasks:

- POS tagging
- Dependency parsing
- Named-entity recognition (NER)
- Natural language inference (NLI)

### Downstream Datasets

| Task | #Train | #Valid | #Test |
|----|-------|-------|------|
| POS tagging | 27,000 | 870 | 2,120 |
| Dependency parsing | 8,977 | 200 | 1,020 |
| NER | 14,861 | 2,000 | 2,831 |
| NLI | 392,702 | 2,490 | 5,010 |

---

### Fine-tuning

- POS & NER: Linear classifier on first subword token
- Dependency parsing: Biaffine parser with PhoBERT contextual embeddings
- NLI: Word-level segmentation before BPE

Hyperparameters:
- Optimizer: AdamW
- Learning rate: 1e-5
- Batch size: 32
- Epochs: up to 30
- Early stopping after 5 epochs without improvement
- Results averaged over 5 runs

---

## 4. Experimental Results

### POS Tagging & Dependency Parsing

| Model | POS Acc (%) | LAS / UAS (%) |
|----|------------|---------------|
| XLM-Rbase | 96.2 | 76.46 / 83.10 |
| XLM-Rlarge | 96.3 | 75.87 / 82.70 |
| **PhoBERTbase** | **96.7** | **78.77 / 85.22** |
| **PhoBERTlarge** | **96.8** | **77.85 / 84.32** |

---

### NER & NLI

| Model | NER F1 (%) | NLI Acc (%) |
|----|-----------|------------|
| XLM-Rbase | 92.0 | 75.4 |
| XLM-Rlarge | 92.8 | 79.7 |
| **PhoBERTbase** | **93.6** | **78.5** |
| **PhoBERTlarge** | **94.7** | **80.0** |

PhoBERT outperforms XLM-R despite using significantly fewer parameters and less pre-training data.

---

## 5. Discussion

- PhoBERTbase slightly outperforms PhoBERTlarge in dependency parsing
- Larger models may not encode syntactic information optimally in the last layer
- Word-level segmentation prior to BPE is crucial for Vietnamese NLP
- Monolingual models outperform multilingual ones even with less data

---

## 6. Conclusion

We introduced **PhoBERT**, the first large-scale monolingual pre-trained language models for Vietnamese. PhoBERT achieves state-of-the-art results on four major NLP tasks and consistently outperforms multilingual models like XLM-R.

By releasing PhoBERT publicly, we aim to foster future Vietnamese NLP research and applications.

---

## References

*(Giữ nguyên toàn bộ danh sách tài liệu tham khảo như bản gốc arXiv/ACL)*

