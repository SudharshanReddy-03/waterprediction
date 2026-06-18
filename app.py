import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from train_model import load_and_prepare_data, train_and_save_models

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

st.set_page_config(
    page_title="Water Quality Predictor",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
        .main-title {
            font-size: 2.2rem;
            font-weight: 700;
            color: #1a6fa3;
            text-align: center;
            margin-bottom: 0.2rem;
        }
        .sub-title {
            font-size: 1rem;
            color: #555;
            text-align: center;
            margin-bottom: 1.5rem;
        }
        .metric-card {
            background-color: #f0f8ff;
            border-left: 5px solid #1a6fa3;
            padding: 1rem;
            border-radius: 8px;
        }
        .good-quality {
            background-color: #d4edda;
            color: #155724;
            padding: 1rem;
            border-radius: 8px;
            font-size: 1.3rem;
            font-weight: bold;
            text-align: center;
        }
        .poor-quality {
            background-color: #f8d7da;
            color: #721c24;
            padding: 1rem;
            border-radius: 8px;
            font-size: 1.3rem;
            font-weight: bold;
            text-align: center;
        }
        .section-header {
            border-bottom: 2px solid #1a6fa3;
            padding-bottom: 4px;
            margin-bottom: 1rem;
            color: #1a6fa3;
        }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_trained_models():
    results, X_train, X_test, y_train, y_test, scaler, feature_cols, df = train_and_save_models()
    return results, X_train, X_test, y_train, y_test, scaler, feature_cols, df


def load_models():
    lr_model = joblib.load(os.path.join(MODELS_DIR, "lr_model.pkl"))
    dt_model = joblib.load(os.path.join(MODELS_DIR, "dt_model.pkl"))
    rf_model = joblib.load(os.path.join(MODELS_DIR, "rf_model.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    feature_cols = joblib.load(os.path.join(MODELS_DIR, "feature_cols.pkl"))
    return lr_model, dt_model, rf_model, scaler, feature_cols


def predict_water_quality(input_values, model_name, lr_model, dt_model, rf_model, scaler, feature_cols):
    input_array = np.array([input_values])
    input_df = pd.DataFrame(input_array, columns=feature_cols)

    if model_name == "Logistic Regression":
        scaled_input = scaler.transform(input_df)
        prediction = lr_model.predict(scaled_input)[0]
        proba = lr_model.predict_proba(scaled_input)[0]
    elif model_name == "Decision Tree":
        prediction = dt_model.predict(input_df)[0]
        proba = dt_model.predict_proba(input_df)[0]
    else:
        prediction = rf_model.predict(input_df)[0]
        proba = rf_model.predict_proba(input_df)[0]

    return prediction, proba


st.markdown('<div class="main-title">💧 AI-Driven Water Quality Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Predict water potability using Machine Learning models trained on real water quality data</div>', unsafe_allow_html=True)

with st.spinner("Loading models and training data..."):
    results, X_train, X_test, y_train, y_test, scaler, feature_cols, df = get_trained_models()
    lr_model, dt_model, rf_model, scaler, feature_cols = load_models()

tabs = st.tabs(["🏠 Overview", "📊 Dataset Analysis", "🤖 Model Evaluation", "🔮 Predict Water Quality"])

with tabs[0]:
    st.markdown("## Project Overview")
    st.markdown("""
    This system uses machine learning to classify water samples as **Good Quality** or **Poor Quality** based on key physicochemical parameters.
    Water quality classification follows standard thresholds:
    - **pH**: 6.5 – 8.5 (safe drinking range)
    - **Dissolved Oxygen (DO)**: ≥ 4 mg/l
    - **Biochemical Oxygen Demand (BOD)**: ≤ 3 mg/l
    """)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        good = (df["target"] == 1).sum()
        st.metric("Good Quality Samples", f"{good:,}")
    with col3:
        poor = (df["target"] == 0).sum()
        st.metric("Poor Quality Samples", f"{poor:,}")
    with col4:
        st.metric("Features Used", len(feature_cols))

    st.markdown("### Features Used for Prediction")
    feat_desc = {
        "temp": "Water Temperature (°C)",
        "do": "Dissolved Oxygen (mg/l)",
        "ph": "pH Level",
        "conductivity": "Conductivity (µmhos/cm)",
        "bod": "Biochemical Oxygen Demand (mg/l)",
        "nitrate": "Nitrate + Nitrite (mg/l)",
        "total_coliform": "Total Coliform (MPN/100ml)",
    }
    feat_df = pd.DataFrame(
        [(k, v) for k, v in feat_desc.items()],
        columns=["Feature Code", "Description"]
    )
    st.table(feat_df)

    st.markdown("### Machine Learning Models")
    model_df = pd.DataFrame({
        "Model": ["Logistic Regression", "Decision Tree", "Random Forest"],
        "Type": ["Linear Classifier", "Tree-based", "Ensemble"],
        "Accuracy": [
            f"{results['Logistic Regression']['accuracy']:.2%}",
            f"{results['Decision Tree']['accuracy']:.2%}",
            f"{results['Random Forest']['accuracy']:.2%}",
        ]
    })
    st.table(model_df)

with tabs[1]:
    st.markdown("## Dataset Analysis")

    st.markdown("### Sample Data")
    display_cols = feature_cols + ["target"]
    show_df = df[display_cols].copy()
    show_df["target"] = show_df["target"].map({1: "Good Quality", 0: "Poor Quality"})
    st.dataframe(show_df.head(20), use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Class Distribution")
        fig, ax = plt.subplots(figsize=(5, 4))
        counts = df["target"].value_counts()
        labels = ["Good Quality", "Poor Quality"]
        colors = ["#28a745", "#dc3545"]
        ax.bar(labels, [counts.get(1, 0), counts.get(0, 0)], color=colors, edgecolor="white", linewidth=1.2)
        ax.set_ylabel("Number of Samples")
        ax.set_title("Water Quality Distribution")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("### Feature Statistics")
        stats_df = df[feature_cols].describe().round(2)
        st.dataframe(stats_df, use_container_width=True)

    st.markdown("### Feature Correlations")
    fig, ax = plt.subplots(figsize=(10, 6))
    corr_data = df[feature_cols + ["target"]].corr()
    sns.heatmap(corr_data, annot=True, fmt=".2f", cmap="Blues", ax=ax, linewidths=0.5)
    ax.set_title("Feature Correlation Matrix")
    st.pyplot(fig)
    plt.close()

    st.markdown("### Feature Distributions by Water Quality")
    selected_feature = st.selectbox("Select a feature to explore", feature_cols)
    fig, ax = plt.subplots(figsize=(8, 4))
    good_vals = df[df["target"] == 1][selected_feature]
    poor_vals = df[df["target"] == 0][selected_feature]
    ax.hist(good_vals, bins=30, alpha=0.6, color="#28a745", label="Good Quality")
    ax.hist(poor_vals, bins=30, alpha=0.6, color="#dc3545", label="Poor Quality")
    ax.set_xlabel(feat_desc.get(selected_feature, selected_feature))
    ax.set_ylabel("Frequency")
    ax.set_title(f"Distribution of {feat_desc.get(selected_feature, selected_feature)}")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    st.pyplot(fig)
    plt.close()

with tabs[2]:
    st.markdown("## Model Evaluation")

    model_choice = st.selectbox(
        "Select model to evaluate",
        ["Logistic Regression", "Decision Tree", "Random Forest"],
        key="eval_model"
    )

    model_result = results[model_choice]

    col1, col2, col3 = st.columns(3)
    report = model_result["report"]
    with col1:
        st.metric("Accuracy", f"{model_result['accuracy']:.4f}")
    with col2:
        st.metric("Precision (Avg)", f"{report['weighted avg']['precision']:.4f}")
    with col3:
        st.metric("Recall (Avg)", f"{report['weighted avg']['recall']:.4f}")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Confusion Matrix")
        fig, ax = plt.subplots(figsize=(5, 4))
        cm = model_result["cm"]
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Poor Quality", "Good Quality"],
            yticklabels=["Poor Quality", "Good Quality"]
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix: {model_choice}")
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("### Classification Report")
        report_df = pd.DataFrame(report).transpose().round(4)
        st.dataframe(report_df, use_container_width=True)

    st.markdown("### Model Accuracy Comparison")
    fig, ax = plt.subplots(figsize=(8, 4))
    model_names = list(results.keys())
    accuracies = [results[m]["accuracy"] for m in model_names]
    colors = ["#1a6fa3" if m == model_choice else "#b0c8e0" for m in model_names]
    bars = ax.bar(model_names, accuracies, color=colors, edgecolor="white", linewidth=1.2)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Comparison of Model Accuracies")
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{acc:.2%}", ha="center", va="bottom", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    st.pyplot(fig)
    plt.close()

    if model_choice == "Random Forest":
        st.markdown("### Feature Importance (Random Forest)")
        importances = rf_model.feature_importances_
        fi_df = pd.DataFrame({"Feature": feature_cols, "Importance": importances})
        fi_df = fi_df.sort_values("Importance", ascending=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(fi_df["Feature"], fi_df["Importance"], color="#1a6fa3")
        ax.set_xlabel("Importance Score")
        ax.set_title("Feature Importances - Random Forest")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        st.pyplot(fig)
        plt.close()

with tabs[3]:
    st.markdown("## Predict Water Quality")
    st.markdown("Enter the water quality parameters below to get a prediction.")

    pred_model = st.selectbox(
        "Choose Prediction Model",
        ["Logistic Regression", "Decision Tree", "Random Forest"],
        key="pred_model"
    )

    st.markdown("### Water Quality Parameters")

    col1, col2 = st.columns(2)

    with col1:
        temp_val = st.slider("Temperature (°C)", 0.0, 50.0, 25.0, 0.1)
        do_val = st.slider("Dissolved Oxygen (mg/l)", 0.0, 20.0, 7.0, 0.1)
        ph_val = st.slider("pH Level", 0.0, 14.0, 7.0, 0.1)
        cond_val = st.slider("Conductivity (µmhos/cm)", 0, 5000, 300, 10)

    with col2:
        bod_val = st.slider("BOD (mg/l)", 0.0, 30.0, 2.0, 0.1)
        nitrate_val = st.slider("Nitrate + Nitrite (mg/l)", 0.0, 50.0, 2.0, 0.1)
        coliform_val = st.slider("Total Coliform (MPN/100ml)", 0, 50000, 100, 50)

    input_values = [temp_val, do_val, ph_val, cond_val, bod_val, nitrate_val, coliform_val]

    st.markdown("### Current Input Summary")
    summary_df = pd.DataFrame({
        "Parameter": [feat_desc[f] for f in feature_cols],
        "Value": input_values
    })
    st.dataframe(summary_df, use_container_width=True)

    if st.button("🔍 Predict Water Quality", type="primary", use_container_width=True):
        prediction, proba = predict_water_quality(
            input_values, pred_model, lr_model, dt_model, rf_model, scaler, feature_cols
        )

        st.markdown("---")
        st.markdown("### Prediction Result")

        if prediction == 1:
            st.markdown(
                '<div class="good-quality">✅ GOOD QUALITY — This water sample meets basic quality standards.</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="poor-quality">⚠️ POOR QUALITY — This water sample does not meet quality standards.</div>',
                unsafe_allow_html=True
            )

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Confidence: Poor Quality", f"{proba[0]:.2%}")
        with col2:
            st.metric("Confidence: Good Quality", f"{proba[1]:.2%}")

        fig, ax = plt.subplots(figsize=(6, 3))
        bar_colors = ["#dc3545", "#28a745"]
        bars = ax.bar(["Poor Quality", "Good Quality"], proba, color=bar_colors, edgecolor="white")
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Probability")
        ax.set_title(f"Prediction Confidence ({pred_model})")
        for bar, p in zip(bars, proba):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{p:.2%}", ha="center", fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        st.pyplot(fig)
        plt.close()

        st.markdown("### Parameter Assessment")
        assessment = []
        ph_status = "✅ Within safe range (6.5-8.5)" if 6.5 <= ph_val <= 8.5 else "❌ Outside safe range"
        do_status = "✅ Adequate (≥ 4 mg/l)" if do_val >= 4 else "❌ Low Dissolved Oxygen"
        bod_status = "✅ Acceptable (≤ 3 mg/l)" if bod_val <= 3 else "❌ High BOD (organic pollution)"
        assessment = pd.DataFrame({
            "Parameter": ["pH Level", "Dissolved Oxygen", "BOD"],
            "Value": [ph_val, do_val, bod_val],
            "Status": [ph_status, do_status, bod_status]
        })
        st.dataframe(assessment, use_container_width=True)

st.markdown("---")
st.markdown(
    "<center><small>Water Quality Prediction System | Machine Learning Project | Built with Streamlit</small></center>",
    unsafe_allow_html=True
)
