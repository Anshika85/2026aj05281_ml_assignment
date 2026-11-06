# Cardiovascular Disease Risk Prediction Portal

**BITS Pilani M.Tech AIML — Machine Learning Assignment 2**

---

## a. Problem statement
Cardiovascular diseases (CVDs) remain the leading cause of global mortality. Early identification of individuals at high risk enables timely clinical interventions and lifestyle modifications, significantly reducing adverse health outcomes. This project builds and deploys an interactive Machine Learning diagnostic pipeline that evaluates patient physiological data to predict cardiovascular disease risk in real time. The goal is to compare five distinct classification algorithms on standard clinical benchmarks and deploy the optimal model via an accessible web portal on Streamlit Community Cloud.

---

## b. Dataset description
- **Dataset Name:** UCI Heart Disease Benchmark Dataset
- **Total Instances:** 920 clinical records
- **Total Features:** 14 attributes (13 input physiological parameters + 1 binary target outcome variable)

### Clinical Feature Breakdown

| Feature No. | Attribute Name | Description & Type | Value Range / Categories |
| :---: | :--- | :--- | :--- |
| 1 | `age` | Patient Age (Continuous) | 29 – 78 years |
| 2 | `sex` | Biological Sex (Binary) | 0 = Female, 1 = Male |
| 3 | `cp` | Chest Pain Type (Categorical) | 0: Typical Angina, 1: Atypical Angina, 2: Non-anginal, 3: Asymptomatic |
| 4 | `trestbps` | Resting Blood Pressure (Continuous) | 94 – 200 mm Hg |
| 5 | `chol` | Serum Cholesterol (Continuous) | 126 – 564 mg/dl |
| 6 | `fbs` | Fasting Blood Sugar > 120 mg/dl (Binary) | 0 = False, 1 = True |
| 7 | `restecg` | Resting ECG Results (Categorical) | 0: Normal, 1: ST-T wave abnormality, 2: LV Hypertrophy |
| 8 | `thalach` | Maximum Heart Rate Achieved (Continuous) | 71 – 202 bpm |
| 9 | `exang` | Exercise-Induced Angina (Binary) | 0 = No, 1 = Yes |
| 10 | `oldpeak` | ST Depression Induced by Exercise (Continuous) | 0.0 – 6.2 |
| 11 | `slope` | Slope of Peak Exercise ST Segment | 0: Upsloping, 1: Flat, 2: Downsloping |
| 12 | `ca` | Major Vessels Colored by Fluoroscopy | 0 – 3 |
| 13 | `thal` | Thalassemia Disorder Status | 1: Normal, 2: Fixed defect, 3: Reversible defect |
| 14 | **`target`** | **Cardiovascular Risk Diagnosis (Label)** | **0 = Low Risk, 1 = Elevated Risk** |

---

## c. Github Repository Link
- **GitHub Repository:** [https://github.com/Anshika85/2026aj05281_ml_assignment](https://github.com/Anshika85/2026aj05281_ml_assignment)
- **Live Streamlit App:** [https://2026aj05281mlassignment-np2fsr9urkdgs5blhwghtk.streamlit.app](https://2026aj05281mlassignment-np2fsr9urkdgs5blhwghtk.streamlit.app)

---

## d. Models used:

### Model Evaluation Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.750 | 0.825 | 0.746 | 0.715 | 0.730 | 0.498 |
| **Decision Tree** | 0.772 | 0.839 | 0.817 | 0.667 | 0.734 | 0.546 |
| **KNN** | 0.740 | 0.815 | 0.743 | 0.690 | 0.715 | 0.478 |
| **Naive Bayes** | 0.746 | 0.818 | 0.742 | 0.708 | 0.725 | 0.489 |
| **Support Vector Machine** | **0.808** | **0.888** | **0.826** | **0.752** | **0.787** | **0.615** |

---

### Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Serves as a solid linear baseline with balanced precision (0.746) and recall (0.715), but struggles to capture complex non-linear feature interactions in clinical data. |
| **Decision Tree** | Achieves high precision (0.817) due to explicit decision boundaries, but suffers from lower recall (0.667), indicating a tendency to miss positive high-risk cases. |
| **KNN** | Demonstrates fair performance (Accuracy: 0.740, AUC: 0.815), but is sensitive to local noise and distance metrics across multi-dimensional physiological spaces. |
| **Naive Bayes** | Performs reliably with an AUC of 0.818 and quick training time, though its conditional independence assumption slightly limits peak predictive precision (0.742). |
| **Support Vector Machine** | **Top Performer.** Using an RBF kernel on standardized features, SVM successfully mapped non-linear clinical boundaries, leading across all metrics (Accuracy: 0.808, AUC: 0.888, F1: 0.787, MCC: 0.615). |
| **Overall Winner for your dataset?** | **Support Vector Machine (SVM)** is the clear overall winner. It achieved the highest overall accuracy (80.8%), the strongest discrimination power (AUC: 0.888), and the highest Matthews Correlation Coefficient (MCC: 0.615), making it the most robust diagnostic engine for cardiovascular risk prediction. |

---

## 🛠️ Local Execution & Setup

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train Models:**
   ```bash
   python train.py
   ```

3. **Launch Streamlit App:**
   ```bash
   streamlit run app.py
   ```
