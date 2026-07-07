# Loan_Approval_Predictor

A machine learning project that predicts whether a loan application will be **Approved** or **Rejected** based on applicant financial and personal details. Built using Python, scikit-learn, and trained on a real-world loan approval dataset.

## Overview

This project uses supervised classification algorithms to analyze applicant data (income, credit score, assets, dependents, etc.) and predict loan approval outcomes. The final model achieves **98.1% accuracy** and a **0.997 AUC score** using a Random Forest Classifier.

## Dataset

The dataset (`loan_approval_dataset.csv`) contains 4,269 loan applications with the following features:

| Feature | Description |
|---|---|
| `no_of_dependents` | Number of people financially dependent on the applicant |
| `education` | Graduate / Not Graduate |
| `self_employed` | Yes / No |
| `income_annum` | Annual income (INR) |
| `loan_amount` | Requested loan amount (INR) |
| `loan_term` | Loan repayment term (years) |
| `cibil_score` | Credit score (300-900) |
| `residential_assets_value` | Value of residential assets |
| `commercial_assets_value` | Value of commercial assets |
| `luxury_assets_value` | Value of luxury assets |
| `bank_asset_value` | Value of bank assets |
| `loan_status` | Target variable - Approved / Rejected |

## Project Structure

```
loan-approval-predictor/
├── loan_approval_dataset.csv       # Dataset
├── loan_approval_predictor.ipynb   # Main notebook (Colab-ready)
├── loan_approval_model.pkl         # Trained Random Forest model
├── scaler.pkl                      # StandardScaler used for preprocessing
├── label_encoders.pkl              # Label encoders for categorical fields
├── confusion_matrix.png            # Model evaluation chart
├── feature_importance.png          # Feature importance chart
└── README.md
```

## Tech Stack

- Python 3.x
- pandas, NumPy
- scikit-learn
- Matplotlib, Seaborn
- Joblib (model serialization)
- Google Colab / Jupyter Notebook

## How It Works

1. **Data Cleaning** - strips whitespace, drops non-predictive `loan_id` column.
2. **Encoding** - categorical fields (`education`, `self_employed`, `loan_status`) encoded using `LabelEncoder`.
3. **Train-Test Split** - 80/20 stratified split.
4. **Scaling** - numeric features standardized using `StandardScaler`.
5. **Model Training** - Logistic Regression, Decision Tree, and Random Forest are trained and compared.
6. **Evaluation** - best model selected based on accuracy and AUC; confusion matrix and feature importance plotted.
7. **Prediction** - trained model, scaler, and encoders saved as `.pkl` files for reuse.

## Model Performance

| Model | Accuracy | AUC |
|---|---|---|
| Logistic Regression | 92.27% | 0.9745 |
| Decision Tree | 97.19% | 0.9937 |
| **Random Forest (Best)** | **98.13%** | **0.9975** |

### Feature Importance

`cibil_score` is by far the most influential feature (~82% importance), followed by `loan_term` and `loan_amount`.

## Getting Started

### Run on Google Colab

1. Open Google Colab: https://colab.research.google.com
2. Upload `loan_approval_predictor.ipynb` or copy the code from the notebook.
3. Run the upload cell and select `loan_approval_dataset.csv` when prompted.
4. Run all cells sequentially (Runtime → Run all).

### Run Locally

```bash
git clone https://github.com/<your-username>/loan-approval-predictor.git
cd loan-approval-predictor
pip install -r requirements.txt
jupyter notebook loan_approval_predictor.ipynb
```

### requirements.txt

```
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib
```

## Making Predictions

```python
import joblib
import pandas as pd

model = joblib.load('loan_approval_model.pkl')
scaler = joblib.load('scaler.pkl')
encoders = joblib.load('label_encoders.pkl')

applicant = pd.DataFrame([{
    'no_of_dependents': 1,
    'education': encoders['education'].transform(['Graduate'])[0],
    'self_employed': encoders['self_employed'].transform(['No'])[0],
    'income_annum': 9000000,
    'loan_amount': 8000000,
    'loan_term': 10,
    'cibil_score': 800,
    'residential_assets_value': 5000000,
    'commercial_assets_value': 3000000,
    'luxury_assets_value': 10000000,
    'bank_asset_value': 6000000
}])

scaled = scaler.transform(applicant)
prediction = model.predict(scaled)
result = encoders['loan_status'].inverse_transform(prediction)[0]

print("Loan Status:", "Selected" if result == "Approved" else "Rejected")
```

## Future Improvements

- Deploy as a REST API using Flask/FastAPI
- Integrate with a MERN stack frontend for a full-stack loan application portal
- Add hyperparameter tuning (GridSearchCV) for improved performance
- Add SHAP/LIME explainability for individual predictions
- Handle class imbalance with SMOTE if dataset is skewed

### Intern_ID: CITS5093
