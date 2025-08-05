# S5E7-TAB-M-0.967
# Introvert vs Extrovert Classification using Tabular Neural Network (TabM)

This notebook tackles a classification problem from a machine learning competition, where the goal is to predict whether a person is an **Introvert** or **Extrovert** based on various features describing their social behavior.

## Objective
Predict the personality type (Introvert/Extrovert) given traits such as:
- Time spent alone
- Stage fear (Yes/No)
- Social event attendance
- Post frequency on social media
- Friends circle size
- Feeling drained after socializing (Yes/No), etc.

## Evaluation Metric
Accuracy Score between predicted and true labels.

## Features
The dataset includes both categorical and numerical features. An additional external dataset (`personality_datasert.csv`) is merged and upsampled to improve generalization.

## Model
A Tabular Neural Network based on `TabM` is used, along with:
- `rtdl-num-embeddings` for numerical feature embeddings
- Class weight balancing
- Training with PyTorch, using F1 and Accuracy as metrics

## Libraries
- `tabm`
- `rtdl-num-embeddings`
- `scikit-learn`, `pandas`, `numpy`, `torch`

## Data Preprocessing
- Categorical encoding (LabelEncoder)
- Numerical scaling (StandardScaler)
- Train/test split
- Class balancing using `compute_class_weight`

## Training Strategy
The notebook includes:
- TabM model setup
- Weighted loss
- Evaluation on test set
- Metrics tracking (F1, Accuracy)

---

Feel free to use this notebook as a starting point for tabular classification tasks with mixed feature types.
