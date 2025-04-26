# 🩺 Diabetes Prediction using XGBoost  

This project aims to build an accurate and reliable machine learning model to predict the likelihood of a person having diabetes based on health indicators. The final deployed model uses the powerful **XGBoost Classifier**, selected after comparing multiple state-of-the-art models.

---

## 📊 Dataset  

- **Source:** [Kaggle - Diabetes_Prediction_Dataset](https://www.kaggle.com/datasets/marshalpatel3558/diabetes-prediction-dataset)  
- **Features:** Age, Gender, Hypertension, Heart Disease, Smoking History, BMI, HbA1c, Fasting Blood Glucose, etc.  
- **Target:** `Diabetes_Status` (0 = Non-Diabetic, 1 = Diabetic)

---

## 📌 Project Workflow  

1. **Data Preprocessing**
   - Missing value handling  
   - Categorical encoding using Label Encoding  
   - Feature scaling using StandardScaler  
   - Created a custom `Diabetes_Status` label using clinical thresholds:
     ```
     If Fasting Blood Glucose > 125 mg/dL OR HbA1c > 6.5 → Diabetic (1)
     Else → Non-Diabetic (0)
     ```

2. **Model Comparison**
   - Logistic Regression  
   - Random Forest  
   - LightGBM  
   - CatBoost  
   - XGBoost ✅ (Best performing model)

3. **Model Evaluation Metrics**
   - Accuracy  
   - Precision  
   - Recall  
   - F1-Score  
   - ROC-AUC  

4. **Model Saving**
   - XGBoost model saved as `.pkl` and `.json`
   - LabelEncoders and Scaler saved as `.pkl`

---

## 📈 Final Model Performance (XGBoost)

| Metric       | Score  |
|:-------------|:---------|
| Accuracy     | 0.9995  |
| Precision    | 0.9994  |
| Recall       | 1.0000  |
| F1-Score     | 0.9997  |
| ROC-AUC      | 1.0000  |

---

## 📦 Files Included  

- `diabetes_prediction.ipynb` – Complete code notebook  
- `xgboost_model.pkl` – Saved trained XGBoost model  
- `xgboost_model.json` – Saved model in JSON format  
- `scaler.pkl` – StandardScaler object  
- `label_encoders.pkl` – Dictionary of LabelEncoders for categorical columns  

---

## 🚀 How to Run  

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Load trained model and scaler:

```python
  import joblib
  model = joblib.load('xgboost_model.pkl')
  scaler = joblib.load('scaler.pkl')
```

3. Predict on new data after preprocessing.

---
