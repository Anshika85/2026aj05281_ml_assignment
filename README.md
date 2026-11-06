# 🩺 Cardiovascular Disease Risk Prediction Portal

**BITS Pilani M.Tech AIML — Machine Learning Assignment 2**

An end-to-end Machine Learning web application deployed on Streamlit Community Cloud that evaluates cardiovascular disease risk using the benchmark **UCI Heart Disease Dataset**.

---

## 📊 Dataset Specifications

- **Source:** Standard UCI Heart Disease Benchmark Dataset
- **Total Instances:** 920 clinical records
- **Features:** 13 input physiological attributes + 1 binary target outcome label (`target`)

### Input Feature Details

| Feature | Attribute Name | Description & Type | Range / Values |
| :--- | :--- | :--- | :--- |
| 1 | `age` | Patient Age (Continuous) | 29 - 78 years |
| 2 | `sex` | Gender (Binary) | 0 = Female, 1 = Male |
| 3 | `cp` | Chest Pain Type (Categorical) | 0: Typical Angina, 1: Atypical Angina, 2: Non-anginal, 3: Asymptomatic |
| 4 | `trestbps` | Resting Blood Pressure (Continuous) | 94 - 200 mm Hg |
| 5 | `chol` | Serum Cholesterol (Continuous) | 126 - 564 mg/dl |
| 6 | `fbs` | Fasting Blood Sugar > 120 mg/dl (Binary) | 0 = False, 1 = True |
| 7 | `restecg` | Resting Electrocardiographic Results | 0: Normal, 1: ST-T wave abnormality, 2: Left ventricular hypertrophy |
| 8 | `thalach` | Maximum Heart Rate Achieved (Continuous) | 71 - 202 bpm |
| 9 | `exang` | Exercise Induced Angina (Binary) | 0 = No, 1 = Yes |
| 10 | `oldpeak` | ST Depression Induced by Exercise (Continuous) | 0.0 - 6.2 |
| 11 | `slope` | Slope of Peak Exercise ST Segment | 0: Upsloping, 1: Flat, 2: Downsloping |
| 12 | `ca` | Number of Major Vessels Colored by Fluoroscopy | 0 - 3 |
| 13 | `thal` | Thalassemia Status | 1: Normal, 2: Fixed defect, 3: Reversible defect |
| 14 | **`target`** | **Cardiovascular Risk Outcome (Label)** | **0 = Low Risk, 1 = Elevated Risk** |

---

## ⚡ Trained Classification Models

The training pipeline (`train.py`) fits a `StandardScaler` on input features and evaluates 5 classifiers:
1. **Support Vector Machine (SVM)** (RBF Kernel)
2. **Logistic Regression**
3. **Decision Tree Classifier**
4. **K-Nearest Neighbors (KNN)**
5. **Gaussian Naive Bayes**

All trained model weights and feature scalers are serialized inside the `model/` directory.

---

## 🛠️ Project Structure

```
.
├── app.py              # Streamlit UI & inference dashboard
├── train.py            # Model training & serialization pipeline
├── test data.csv       # 920-instance UCI Heart Disease dataset
├── requirements.txt    # Python dependency pinning
├── README.md           # Project documentation
└── model/              # Model artifacts directory
    ├── scaler.pkl
    ├── support_vector_machine.pkl
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    └── naive_bayes.pkl
```

---

## 🚀 Local Setup & Run

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train Models:**
   ```bash
   python train.py
   ```

3. **Launch Web Application:**
   ```bash
   streamlit run app.py
   ```
