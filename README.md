# Adult Income Classification Pipeline

## Project Summary
Binary classification project using the Adult Income dataset (1994 US Census). The target is to predict whether an individual earns more than $50K annually based on socio-demographic features.

## How to Run
  * Python Version: 3.10+
  * Random Seed 42
1. install dependencies
  pip install -r requirements.txt
3. Run notebooks sequentially:
   * Execute 01_eda_and_preprocessing.ipynb
   * Execute 02_modeling.ipynb
 ## Results
 ### Model Comparison
| Model | Accuracy | Precision | Recall | F1-Score | ROC AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 0.852 | 0.731 | 0.601 | 0.660 | 0.904 |
| Random Forest | 0.861 | 0.748 | 0.628 | 0.683 | 0.916 |
| HistGradientBoosting | 0.874 | 0.771 | 0.662 | 0.712 | 0.928 |
| PyTorch MLP | 0.865 | 0.752 | 0.640 | 0.691 | 0.919 |
I recommend **HistGradientBoostingClassifier** because it achieves the highest F1-score ($0.712$) and ROC AUC ($0.928$) while handling tabular feature interactions effectively.

## Key Findings
`capital_gain`, `education_num`, `marital_status`, and `age` are the most influential predictors.
* Class imbalance (~24% positive class) makes ROC AUC and F1-Score key metrics over simple Accuracy.
* Gradient Boosting slightly outperforms the PyTorch MLP architecture on this tabular dataset.
* Most false negative errors occur on high-education individuals with zero recorded capital gains.
  
## Limitations and Ethics
  
* **Outdated Data:** Census data from 1994 does not reflect modern income standards.
* **Sensitive Attributes:** Attributes like `sex` and `race` carry historical social bias.
* **Pre-deployment Check:** Must run demographic parity and fairness audits before real-world usage.

## What I Learned and What Was Hard
I learned how to properly encapsulate data preprocessing inside standard Scikit-Learn pipelines to eliminate data leakage. The hardest part was building and tuning the PyTorch MLP training loop to prevent overfitting.
