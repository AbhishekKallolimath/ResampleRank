# Dataset Profiling

## 1. Overview

Dataset profiling is the first stage of the ResampleRank framework.

The purpose of this stage is to describe the characteristics of an imbalanced tabular dataset before applying any resampling technique. These characteristics are later used as meta-features for the meta-learning component.

The current implementation reads a feature dataset and its target labels, calculates basic dataset characteristics, and stores the resulting profile as a CSV file.

## 2. Current Dataset

The first dataset used for testing is the Pima Indians Diabetes dataset.

| Characteristic          |           Value |
| ----------------------- | --------------: |
| Samples                 |             768 |
| Features                |               8 |
| Numeric Features        |               8 |
| Categorical Features    |               0 |
| Missing Values          |               0 |
| Duplicate Rows          |               0 |
| Majority Class          | tested_negative |
| Minority Class          | tested_positive |
| Majority Samples        |             500 |
| Minority Samples        |             268 |
| Minority Proportion     |           0.349 |
| Imbalance Ratio         |           1.866 |
| Feature-to-Sample Ratio |          0.0104 |

## 3. Meta-Features

The profiler currently extracts the following dataset-level characteristics:

### Number of Samples

The total number of observations in the dataset.

### Number of Features

The total number of input features used for classification.

### Numeric Features

The number of features containing numeric data.

### Categorical Features

The number of features containing non-numeric data.

### Missing Values

The total number of missing values present in the input feature dataset.

### Duplicate Rows

The number of duplicated observations in the feature dataset.

### Majority-Class Count

The number of samples belonging to the majority class.

### Minority-Class Count

The number of samples belonging to the minority class.

### Minority Proportion

The proportion of samples belonging to the minority class:

$$
Minority\ Proportion =
\frac{N_{minority}}{N_{total}}
$$

### Imbalance Ratio

The ratio between the majority-class size and minority-class size:

$$
Imbalance\ Ratio =
\frac{N_{majority}}{N_{minority}}
$$

A larger value indicates stronger class imbalance.

### Feature-to-Sample Ratio

The ratio between the number of features and the number of samples:

$$
Feature\text{-}Sample\ Ratio =
\frac{N_{features}}{N_{samples}}
$$

This provides a simple description of the dimensionality of the dataset relative to its sample size.

## 4. Implementation

The profiling functionality is implemented in:

```text
src/dataset_profiler.py
```

The profiler accepts two files:

```text
data/raw/pima.csv
data/raw/pima_target.csv
```

The generated profile is stored in:

```text
data/processed/dataset_profiles.csv
```

## 5. Workflow

The current profiling workflow is:

```text
Raw Dataset
     ↓
Load Features and Target
     ↓
Identify Class Distribution
     ↓
Calculate Dataset Characteristics
     ↓
Generate Meta-Features
     ↓
Save Dataset Profile
```

## 6. Purpose in ResampleRank

The dataset profile will later become an input to the meta-learning system.

The overall research workflow is:

```text
Dataset
   ↓
Dataset Profiling
   ↓
Meta-Features
   ↓
Resampling Experiments
   ↓
Classification Performance
   ↓
Meta-Dataset
   ↓
Meta-Learning Model
   ↓
Prediction and Ranking of Resampling Strategies
```

The current profiler therefore provides the foundation for the later prediction and ranking stages of ResampleRank.

## 7. Future Extensions

The current profiler contains basic structural and imbalance characteristics. As the project develops, additional dataset-level characteristics may be added, such as:

* feature variance and dispersion
* class overlap measures
* feature correlation
* dimensionality and sparsity
* class complexity measures
* nearest-neighbour based characteristics
* measures related to minority-class difficulty

These additional characteristics will be evaluated based on their usefulness for predicting resampling effectiveness.
