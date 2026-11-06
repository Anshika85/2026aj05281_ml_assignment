import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

st.set_page_config(
    page_title="Cardiovascular Risk Predictor",
    page_icon="🩺",
    layout="wide"
)

# Custom header styling
st.title("🩺 Cardiovascular Disease Risk Inference Portal")
st.caption("BITS Pilani M.Tech AIML — Machine Learning Assignment 2")
st.markdown("---")

st.sidebar.title("🎛️ Diagnostics Panel")
uploaded_csv = st.sidebar.file_uploader("Upload Patient Test Batch (CSV)", type=["csv"])

chosen_classifier = st.sidebar.selectbox(
    "Select ML Classification Engine",
    ["Support Vector Machine", "Logistic Regression", "Decision Tree", "KNN", "Naive Bayes"]
)

model_path_lookup = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "KNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Support Vector Machine": "model/support_vector_machine.pkl"
}

if uploaded_csv is not None:
    df_input = pd.read_csv(uploaded_csv)
    
    st.subheader("📌 Patient Clinical Data Preview")
    st.dataframe(df_input.head(5), use_container_width=True)
    
    if "target" not in df_input.columns:
        st.error("Validation Error: Uploaded dataset must contain a 'target' column.")
    else:
        X_test_df = df_input.drop("target", axis=1)
        y_test_true = df_input["target"]
        
        try:
            scaler_model = joblib.load("model/scaler.pkl")
            model_engine = joblib.load(model_path_lookup[chosen_classifier])
            
            X_scaled_eval = scaler_model.transform(X_test_df)
            
            y_pred_vals = model_engine.predict(X_scaled_eval)
            y_prob_vals = model_engine.predict_proba(X_scaled_eval)[:, 1] if hasattr(model_engine, "predict_proba") else y_pred_vals
            
            acc = accuracy_score(y_test_true, y_pred_vals)
            auc = roc_auc_score(y_test_true, y_prob_vals)
            prec = precision_score(y_test_true, y_pred_vals, zero_division=0)
            rec = recall_score(y_test_true, y_pred_vals, zero_division=0)
            f1 = f1_score(y_test_true, y_pred_vals, zero_division=0)
            mcc = matthews_corrcoef(y_test_true, y_pred_vals)
            
            st.subheader(f"⚡ Performance Metrics: {chosen_classifier}")
            
            grid_c1, grid_c2, grid_c3, grid_c4, grid_c5, grid_c6 = st.columns(6)
            grid_c1.metric("Accuracy", f"{acc:.3f}")
            grid_c2.metric("AUC", f"{auc:.3f}")
            grid_c3.metric("Precision", f"{prec:.3f}")
            grid_c4.metric("Recall", f"{rec:.3f}")
            grid_c5.metric("F1 Score", f"{f1:.3f}")
            grid_c6.metric("MCC Score", f"{mcc:.3f}")
            
            st.markdown("---")
            col_viz_left, col_viz_right = st.columns(2)
            
            with col_viz_left:
                st.subheader("🎯 Diagnostic Confusion Matrix")
                cm_data = confusion_matrix(y_test_true, y_pred_vals)
                fig_cm, ax_cm = plt.subplots(figsize=(5, 3.8))
                sns.heatmap(cm_data, annot=True, fmt="d", cmap="Teal", ax=ax_cm, cbar=False)
                plt.xlabel("Predicted Risk Status")
                plt.ylabel("Actual Risk Status")
                st.pyplot(fig_cm)
                
            with col_viz_right:
                st.subheader("📋 Classification Details")
                clf_summary = classification_report(y_test_true, y_pred_vals, output_dict=False)
                st.code(clf_summary)
                
        except Exception as err:
            st.error(f"Inference error encountered: {err}")
else:
    st.info("👈 Please upload `test data.csv` from the sidebar to inspect diagnostic evaluation results.")
