# ResampleRank

### Meta-Learning for Predicting Resampling Effectiveness in Imbalanced Tabular Classification

ResampleRank is a research project that investigates whether the characteristics of an imbalanced tabular dataset can be used to predict and rank the effectiveness of different resampling strategies.

The project combines conventional imbalanced classification experiments with dataset-level meta-learning. The goal is to study whether knowledge obtained from previously evaluated datasets can generalize to completely unseen datasets.

---

## Research Objective

Class imbalance occurs when one class contains significantly fewer observations than another. This can cause machine learning models to perform poorly on the minority class.

Several resampling techniques can be used to address this problem, including:

- Random Oversampling
- Random Undersampling
- SMOTE
- ADASYN
- SMOTE-Tomek
- SMOTE-ENN

However, the effectiveness of these techniques can vary depending on the characteristics of the dataset.

ResampleRank investigates whether a meta-learning model can learn this relationship and predict which resampling strategies are likely to perform well on a previously unseen dataset.

---

## Research Problem

The central problem investigated by this project is:

> Can dataset-level characteristics be used to predict and rank the effectiveness of different resampling strategies for unseen imbalanced tabular classification datasets?

The study evaluates the relationship between:

```text
Dataset Characteristics
        +
Resampling Strategy
        +
Classifier
        ↓
Observed Classification Performance