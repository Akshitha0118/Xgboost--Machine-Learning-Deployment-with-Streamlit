# =========================================================
# 1. IMPORT LIBRARIES
# =========================================================
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, roc_auc_score, ConfusionMatrixDisplay

# =========================================================
# 2. PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="XGBoost Bank Churn Dashboard",
    page_icon="📊",
    layout="wide"
)

# =========================================================
# 3. CUSTOM CSS
# =========================================================
st.markdown("""
<style>
body {
    background-color: #f8fafc;   /* light app background */
    color: white;                /* unchanged */
}

.title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    color: #38bdf8;               /* unchanged */
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #cbd5e1;               /* unchanged */
}

.card {
    background-color: #ffffff;   /* light card */
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.08);
}

.metric-box {
    background-color: #e5e7eb;   /* light metric box */
    padding: 15px;
    border-radius: 12px;
    text-align: center;
}            
</style>
""", unsafe_allow_html=True)


# =========================================================
# 4. TITLE
# =========================================================
st.markdown('<div class="title">📊 XGBoost Churn Prediction Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Bank Customer Churn Dataset (Visualization)</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# =========================================================
# 5. LOAD MODEL & ENCODERS
# =========================================================
with open("xgboost_churn_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoder_gender.pkl", "rb") as f:
    le = pickle.load(f)

with open("column_transformer.pkl", "rb") as f:
    ct = pickle.load(f)

# =========================================================
# 6. LOAD DATASET (SAMPLE DATA)
# =========================================================
dataset = pd.read_csv("Churn_Modelling.csv")

X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

# Encode Gender
X[:, 2] = le.transform(X[:, 2])

# Apply OneHotEncoding
X = ct.transform(X)

dataset = pd.read_csv("Churn_Modelling.csv")

# ===================== DATASET DISPLAY ==================
st.markdown("## 📄 Dataset Overview")

col_a, col_b = st.columns(2)

with col_a:
    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
    st.metric("Total Rows", dataset.shape[0])
    st.markdown('</div>', unsafe_allow_html=True)

with col_b:
    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
    st.metric("Total Columns", dataset.shape[1])
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("### 🔍 Sample Records")
st.dataframe(dataset.head(10), use_container_width=True)

with st.expander("📂 Show Full Dataset"):
    st.dataframe(dataset, use_container_width=True)


# =========================================================
# 7. MODEL PREDICTIONS
# =========================================================
y_pred = model.predict(X)
y_prob = model.predict_proba(X)[:, 1]

accuracy = accuracy_score(y, y_pred)
bias = model.score(X, y)
cm = confusion_matrix(y, y_pred)

# =========================================================
# 8. METRICS
# =========================================================
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
    st.metric("Accuracy", f"{accuracy:.2f}")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
    st.metric("Bias (Train Score)", f"{bias:.2f}")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
    st.metric("Model Used", "XGBoost")
    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# 9. CONFUSION MATRIX
# =========================================================
st.markdown("### 📊 Confusion Matrix")

fig, ax = plt.subplots()
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Churn", "Churn"])
disp.plot(cmap="Blues", ax=ax, values_format="d")
st.pyplot(fig)

# =========================================================
# 10. ROC CURVE
# =========================================================
st.markdown("### 📈 ROC Curve")

fpr, tpr, _ = roc_curve(y, y_prob)
auc_score = roc_auc_score(y, y_prob)

fig, ax = plt.subplots()
ax.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
ax.plot([0, 1], [0, 1], linestyle="--")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend()
st.pyplot(fig)

# =========================================================
# 11. ACTUAL vs PREDICTED
# =========================================================
st.markdown("### 🔁 Actual vs Predicted (First 50 Samples)")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(y[:50], label="Actual", marker="o")
ax.plot(y_pred[:50], label="Predicted", marker="x")
ax.set_xlabel("Sample Index")
ax.set_ylabel("Churn")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# =========================================================
# 12. BIAS vs ACCURACY
# =========================================================
st.markdown("### ⚖️ Bias vs Accuracy")

fig, ax = plt.subplots()
ax.bar(["Bias", "Accuracy"], [bias, accuracy])
ax.set_ylim(0, 1)
ax.set_ylabel("Score")
st.pyplot(fig)

# =========================================================
# 13. FOOTER
# =========================================================
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<center>🚀 Streamlit Dashboard | XGBoost Churn Analysis</center>",
    unsafe_allow_html=True
)
