# UIT-VSFC: Vietnamese Students’ Feedback Corpus for Sentiment Analysis

**Kiet Van Nguyen**, **Vu Duc Nguyen**, **Phu X. V. Nguyen**, **Tham T. H. Truong**, **Ngan Luu-Thuy Nguyen**  
University of Information Technology  
Vietnam National University - Ho Chi Minh City, Vietnam  

Emails:  
- kietnv@uit.edu.vn  
- 14521100@gm.uit.edu.vn  
- 14520685@gm.uit.edu.vn  
- 14520824@gm.uit.edu.vn  
- ngannlt@uit.edu.vn  

Proceedings of the **10th International Conference on Knowledge and Systems Engineering (KSE 2018)**  
©2018 IEEE

---

## Abstract

Students’ feedback is a vital resource for interdisciplinary research combining sentiment analysis and education. To strengthen Vietnamese sentiment analysis, a low-resource language, we build a Vietnamese Students’ Feedback Corpus (UIT-VSFC), a free and high-quality corpus for sentiment-based and topic-based classification tasks.

The corpus contains over 16,000 sentences annotated for sentiment and topic. Annotation consistency and accuracy are ensured through detailed guidelines. Inter-annotator agreement reaches **91.20%** for sentiment classification and **71.07%** for topic classification. Baseline experiments using Maximum Entropy achieve **87.94% F1-score** for sentiment classification and **84.03% F1-score** for topic classification.

These results demonstrate that UIT-VSFC is a reliable and valuable resource for sentiment analysis and educational research.

---

## I. Introduction

Students’ feedback is both a powerful source for improving education quality and an important resource for sentiment analysis research. However, Vietnamese remains a low-resource language in NLP, and education-domain sentiment corpora are scarce.

The UIT-VSFC corpus was created with two motivations:

1. Address the lack of publicly available Vietnamese NLP corpora.
2. Support education quality improvement through automated feedback analysis.

The corpus supports two tasks: sentiment-based classification and topic-based classification. Annotation guidelines were carefully designed to ensure consistency and accuracy. The corpus quality is evaluated using inter-annotator agreement and classification experiments.

### Contributions

1. A publicly available Vietnamese students’ feedback corpus with over 16,000 annotated sentences.
2. Analysis of Vietnamese students’ feedback characteristics and annotation challenges.

---

## II. Students’ Feedback and Sentiment Analysis

Students’ feedback significantly affects learning outcomes, either positively or negatively [1]. Two types of feedback exist:

1. Lecturer-to-student feedback
2. Student-to-lecturer feedback

Research on sentiment analysis has evolved from news articles to social media platforms such as Twitter (SemEval 2013–2017). However, sentiment analysis in education and Vietnamese language contexts remains underexplored.

Existing Vietnamese sentiment research focuses mainly on e-commerce domains (e.g., product reviews, hotel reviews). This motivates the creation of UIT-VSFC for education-related sentiment analysis.

---

## III. Description of Vietnamese Students’ Feedback

Vietnamese students’ feedback consists of short, informal messages containing:

- Acronyms (e.g., `gv` – lecturer, `sv` – student)
- Spelling errors
- Emojis and emotion symbols (`:D`, `:v`, `:(`)
- Mixed sentiment expressions

Feedback topics include lecturers, curriculum, facilities, and miscellaneous issues.

---

## IV. Task Definition and Annotation Guidelines

### A. Task Definition

#### 1. Sentiment-Based Task
Each sentence is labeled as:
- **Positive**
- **Negative**
- **Neutral**

#### 2. Topic-Based Task
Each sentence is labeled as:
- **Lecturer**
- **Curriculum**
- **Facility**
- **Others**

#### Example Annotations

| No. | Sentence | Sentiment | Topic |
|----|---------|----------|------|
| 1 | Giảng viên hướng dẫn tận tình và chu đáo. | Positive | Lecturer |
| 2 | Nội dung môn học chưa đủ và chưa đúng với đề cương. | Negative | Curriculum |
| 3 | Nhà trường cần cải thiện hệ thống điện và máy chiếu. | Negative | Facility |
| 4 | Em không có bất cứ một lời phê bình nào. | Neutral | Others |

---

### B. Annotation Guidelines

#### 1. Sentiment-Based Annotation

- **Positive**: Satisfaction, compliments
- **Negative**: Complaints, dissatisfaction, requests
- **Neutral**: Objective or unclear statements

For mixed-sentiment sentences, annotators choose the stronger polarity.

#### 2. Topic-Based Annotation

- **Lecturer**: Teaching methods, attitudes, expertise
- **Curriculum**: Subjects, assignments, exams, syllabus
- **Facility**: Equipment, infrastructure
- **Others**: Unclear or unrelated topics

---

## V. Corpus Building Process

### A. Data Collection

Feedback was collected from a Vietnamese university between 2013–2016.

| Academic Year | Lecturers | Students | Subjects | Responses |
|--------------|----------|----------|---------|-----------|
| 2014–2015 | 175 | 2,235 | 143 | 6,038 |
| 2015–2016 | 184 | 2,856 | 160 | 6,288 |
| 2016–2017 | 227 | 3,789 | 175 | 13,417 |

---

### B. Pre-processing

Steps include:
- Sentence segmentation
- Abbreviation expansion
- Spelling correction
- Anonymization of personal information

---

### C. Annotation Process

Annotators followed detailed guidelines and discussed difficult cases. Guidelines were iteratively updated, and annotators were retrained when necessary.

---

### D. Resulted Corpus

#### Sentiment and Topic Distribution (%)

| Topic | Positive | Negative | Neutral | Total |
|------|---------|----------|--------|------|
| Lecturer | 33.57 | 25.38 | 1.81 | 71.76 |
| Curriculum | 3.40 | 14.39 | 1.00 | 18.79 |
| Facility | 0.11 | 4.21 | 0.08 | 4.40 |
| Others | 1.61 | 2.01 | 1.43 | 5.04 |
| **Total** | **49.69** | **45.99** | **4.32** | **100** |

---

## VI. Corpus Evaluation

### A. Inter-Annotator Agreement

Agreement measure \( A_m \):

\[
A_m = \frac{P_0 - P_e}{1 - P_e}
\]

| Task | P₀ | Pₑ | Aₘ |
|----|----|----|----|
| Sentiment-based | 95.27 | 46.19 | 91.20 |
| Topic-based | 90.33 | 66.59 | 71.07 |

---

### B. Experimental Results

Classifiers:
- Naive Bayes
- Maximum Entropy (MaxEnt)

| Task | Naive Bayes | MaxEnt |
|----|------------|-------|
| Sentiment-based | 86.1 | **87.9** |
| Topic-based | 83.0 | **84.0** |

---

## VII. Analysis

- Performance improves rapidly with first 1,000 sentences.
- MaxEnt consistently outperforms Naive Bayes.
- Topic confusion is highest in the **Curriculum** category.

---

## VIII. Conclusion and Future Work

We introduced UIT-VSFC, a Vietnamese sentence-level sentiment corpus with high annotation reliability. Baseline experiments achieved strong performance using traditional machine learning models.

Future work includes:
- Error analysis and re-annotation
- Document-level corpus construction
- Deep learning models for sentiment and topic classification

---

## References

[1] John Hattie and Helen Timperley, *The Power of Feedback*, Review of Educational Research, 2007.  
[2]–[21] *(Giữ nguyên danh sách tham khảo như bản gốc IEEE)*

