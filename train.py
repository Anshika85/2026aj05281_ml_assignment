import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

os.makedirs("model", exist_ok=True)

np.random.seed(999)
n_samples = 920

health_features = [
    "patient_age", "systolic_bp", "diastolic_bp", "cholesterol_mg_dl",
    "fasting_blood_sugar", "bmi_index", "resting_heart_rate", "exercise_hours_weekly",
    "smoking_pack_years", "family_history_score", "daily_sodium_intake",
    "hba1c_level", "stress_index", "sleep_duration_hours"
]

X_mat = np.random.randn(n_samples, 14)
y_raw = (
    X_mat[:, 0] * 0.50 + X_mat[:, 1] * 0.45 + X_mat[:, 3] * 0.40 
    + X_mat[:, 5] * 0.35 + X_mat[:, 8] * 0.30 + X_mat[:, 11] * 0.25 
    - X_mat[:, 7] * 0.20 + np.random.randn(n_samples) * 0.35 > 0.15
).astype(int)

df = pd.DataFrame(X_mat, columns=health_features)
df["target"] = y_raw

X = df.drop("target", axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=999)

test_df = X_test.copy()
test_df["target"] = y_test
test_df.to_csv("test data.csv", index=False)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, "model/scaler.pkl")

models_map = {
    "logistic_regression": LogisticRegression(C=0.8, max_iter=300, random_state=999),
    "decision_tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=3, random_state=999),
    "knn": KNeighborsClassifier(n_neighbors=9, p=1),
    "naive_bayes": GaussianNB(var_smoothing=1e-7),
    "support_vector_machine": SVC(C=2.5, gamma="scale", probability=True, random_state=999)
}

for name, model_inst in models_map.items():
    model_inst.fit(X_train_scaled, y_train)
    joblib.dump(model_inst, f"model/{name}.pkl")

print("Training finished successfully. Saved custom artifacts.")
