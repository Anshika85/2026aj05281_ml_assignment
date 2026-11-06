import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import joblib
import os

def train_and_save_models():
    # 1. Load dataset from local test data.csv
    data_path = 'test data.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Error: {data_path} not found in current directory.")
        
    df = pd.read_csv(data_path)
    
    if 'target' not in df.columns:
        raise ValueError("Dataset must contain a 'target' column.")
        
    X = df.drop('target', axis=1)
    y = df['target']
    
    # 2. Split dataset into Train and Validation sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # 3. Fit Standard Scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # 4. Initialize classifiers
    models = {
        'support_vector_machine': SVC(probability=True, kernel='rbf', C=1.0, random_state=42),
        'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
        'decision_tree': DecisionTreeClassifier(max_depth=5, random_state=42),
        'knn': KNeighborsClassifier(n_neighbors=5),
        'naive_bayes': GaussianNB()
    }
    
    # 5. Create output model directory
    os.makedirs('model', exist_ok=True)
    
    # 6. Save scaler
    joblib.dump(scaler, 'model/scaler.pkl')
    print("Saved feature scaler -> model/scaler.pkl")
    
    # 7. Train and save models
    for model_name, model_obj in models.items():
        model_obj.fit(X_train_scaled, y_train)
        save_path = f'model/{model_name}.pkl'
        joblib.dump(model_obj, save_path)
        print(f"Trained & saved {model_name} -> {save_path}")

if __name__ == '__main__':
    train_and_save_models()
