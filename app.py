import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, make_scorer
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import os

# Set page config
st.set_page_config(page_title="UFC KO Prediction", page_icon="🥊", layout="wide")

# Title
st.title("🥊 UFC Fight Outcome Prediction")
st.markdown("Predict KO/TKO vs Non-KO outcomes based on round-by-round UFC Lightweight division statistics.")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("processed-dataset/ufc_df.csv")
    return df

df = load_data()

# Sample data as in randomforest.py
ko_100 = df[df["ko여부"] == 1].sample(100, random_state=42)
nko_100 = df[df["ko여부"] == 0].sample(100, random_state=42)
df_200 = pd.concat([ko_100, nko_100]).sample(frac=1, random_state=42)

# Final columns from randomforest.py
final_cols = ['final_round', 'winner_clinch_succ', 'winner_distance_succ', 'winner_sig_str_succ', 'winner_head_succ', 'winner_body_att']

X = df_200[final_cols]
y = df_200["ko여부"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1, stratify=y
)

# Train model
@st.cache_resource
def train_model():
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=4,
        min_samples_split=5,
        min_samples_leaf=5,
        random_state=1,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    return rf

rf_model = train_model()

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Data Overview", "🎯 Model Performance", "🔮 Prediction"])

with tab1:
    st.header("Data Overview")
    st.write("Sample of the dataset:")
    st.dataframe(df.head(10))
    
    st.subheader("KO Distribution")
    fig, ax = plt.subplots()
    df["ko여부"].value_counts().plot(kind="bar", ax=ax)
    ax.set_title("KO Distribution")
    ax.set_xlabel("KO 여부")
    ax.set_ylabel("Count")
    st.pyplot(fig)

with tab2:
    st.header("Model Performance")
    
    # Test predictions
    y_test_pred = rf_model.predict(X_test)
    y_test_proba = rf_model.predict_proba(X_test)[:, 1]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Classification Report")
        report = classification_report(y_test, y_test_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())
    
    with col2:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_test_pred)
        cm_df = pd.DataFrame(
            cm,
            index=["Actual Non-KO", "Actual KO"],
            columns=["Predicted Non-KO", "Predicted KO"],
        )
        st.dataframe(cm_df)
    
    # ROC Curve
    st.subheader("ROC Curve")
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    auc_score = roc_auc_score(y_test, y_test_proba)
    
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    st.pyplot(fig)
    
    # Feature Importance
    st.subheader("Feature Importance")
    importances = rf_model.feature_importances_
    feat_df = pd.DataFrame({"Feature": final_cols, "Importance": importances}).sort_values("Importance", ascending=False)
    st.bar_chart(feat_df.set_index("Feature"))

with tab3:
    st.header("Make a Prediction")
    st.write("Enter fight statistics to predict if it will end in KO/TKO.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        final_round = st.slider("Final Round", 1, 5, 3)
        winner_clinch_succ = st.number_input("Winner Clinch Successful", 0, 100, 5)
        winner_distance_succ = st.number_input("Winner Distance Successful", 0, 100, 20)
    
    with col2:
        winner_sig_str_succ = st.number_input("Winner Significant Strikes Successful", 0, 100, 25)
        winner_head_succ = st.number_input("Winner Head Strikes Successful", 0, 100, 15)
        winner_body_att = st.number_input("Winner Body Strike Attempts", 0, 100, 10)
    
    with col3:
        if st.button("Predict"):
            input_data = np.array([[final_round, winner_clinch_succ, winner_distance_succ, winner_sig_str_succ, winner_head_succ, winner_body_att]])
            prediction = rf_model.predict(input_data)[0]
            proba = rf_model.predict_proba(input_data)[0]
            
            if prediction == 1:
                st.success(f"Predicted: KO/TKO (Probability: {proba[1]:.2f})")
            else:
                st.info(f"Predicted: Non-KO (Probability: {proba[0]:.2f})")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit | Data from UFCStats.com")