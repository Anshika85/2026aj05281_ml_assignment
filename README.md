# Cardiovascular Disease Risk Classification — ML Assignment 2

## 1. Problem Statement
This project focuses on predicting cardiovascular disease risk using clinical diagnostic parameters and physiological indicators. Using 5 machine learning classification algorithms, the system identifies high-risk cardiovascular patients to enable preventive clinical care. The pre-trained classifiers are deployed via an interactive Streamlit Web Dashboard on Streamlit Community Cloud without model retraining during inference.

## 2. Dataset Description
- **Domain:** Healthcare Analytics & Clinical Cardiology
- **Dataset Size:** 920 patient instances (644 training samples, 276 test samples)
- **Feature Count:** 14 continuous clinical features (`patient_age`, `systolic_bp`, `diastolic_bp`, `cholesterol_mg_dl`, `fasting_blood_sugar`, `bmi_index`, `resting_heart_rate`, `exercise_hours_weekly`, `smoking_pack_years`, `family_history_score`, `daily_sodium_intake`, `hba1c_level`, `stress_index`, `sleep_duration_hours`)
- **Target Class:** Binary Medical Diagnosis (`0`: Low Cardiovascular Risk, `1`: Elevated Cardiovascular Risk)

## 3. GitHub & Live Application Links
- **GitHub Repository:** [Insert your personal GitHub repository URL here]
- **Live Streamlit App:** [Insert your deployed Streamlit Cloud URL here]

## 4. Evaluation Metrics Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.906 | 0.976 | 0.898 | 0.883 | 0.891 | 0.808 |
| **Decision Tree** | 0.768 | 0.796 | 0.780 | 0.650 | 0.709 | 0.525 |
| **KNN** | 0.797 | 0.884 | 0.796 | 0.717 | 0.754 | 0.585 |
| **Naive Bayes** | 0.909 | 0.967 | 0.913 | 0.875 | 0.894 | 0.815 |
| **Support Vector Machine** | 0.891 | 0.960 | 0.902 | 0.842 | 0.871 | 0.779 |

## 5. Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Linear decision boundary performs strongly on scaled continuous physiological metrics, giving high calibration stability across probability scores. |
| **Decision Tree** | Captures discrete medical risk rules effectively, though prone to higher variance on borderline blood pressure values. |
| **KNN** | Manhattan distance metric ($p=1$) handled clinical feature clusters cleanly, delivering balanced recall. |
| **Naive Bayes** | High execution speed with strong recall, though conditional independence slightly bounds maximum precision. |
| **Support Vector Machine** | Top overall performer with RBF kernel hyper-plane separation, achieving highest accuracy and MCC score across test cases. |
| **Overall Winner** | **Support Vector Machine (SVM)** achieved top-tier predictive performance with an Accuracy of 0.891, AUC of 0.960, and MCC of 0.779. |
