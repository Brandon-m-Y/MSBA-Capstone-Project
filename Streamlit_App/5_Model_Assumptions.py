import streamlit as st

st.header("About the Models and Their Assumptions")
st.caption(
    "Three classifiers were trained and evaluated on 39,803 records. "
    "Tuned XGBoost was selected as the production model based on PR-AUC."
)
st.divider()

IMG = "Images/Classification_Models_Images"

tab_overview, tab_xgb, tab_rf, tab_svm, tab_limits = st.tabs([
    "Model Comparison",
    "Tuned XGBoost",
    "Tuned Random Forest",
    "SVM Baseline",
    "Methodology & Limitations",
])

# ── Model Comparison ───────────────────────────────────────────────────────────
with tab_overview:
    st.subheader("Performance Summary")
    st.markdown(
        "**PR-AUC (Average Precision)** is the primary metric. It is directly sensitive to the "
        "minority class and does not inflate in imbalanced settings the way ROC-AUC can. "
        "Random chance baseline PR-AUC for this dataset is ~0.113 (the positive class rate)."
    )
    st.markdown(
        """
        | Model | CV PR-AUC | Notes |
        |-------|:---------:|-------|
        | SVM (RBF, balanced) | Baseline | Good boundary modeling but outperformed by ensembles |
        | Tuned Random Forest | 0.8277 | Strong, competitive; slower inference |
        | **Tuned XGBoost** | **0.8522** | **Selected — highest PR-AUC, fast inference** |
        """
    )

    st.subheader("Train / Test Split")
    st.markdown(
        """
        - **Split:** 80% train / 20% test with **stratified sampling** to preserve the 11.3% positive-class ratio
        - **Train set:** 31,842 rows | **Test set:** 7,961 rows
        - `last_contact_duration_sec` was **excluded** — it is only observable *after* a call
          and would constitute data leakage in a pre-call scoring system
        """
    )

    st.subheader("Imbalance Handling")
    st.markdown(
        """
        Class weighting was used throughout (no SMOTE or undersampling):
        - **Random Forest:** `class_weight='balanced'` and `'balanced_subsample'` variants
        - **XGBoost:** `scale_pos_weight` = ratio of negative to positive samples (~7.8)

        This preserves all training data while penalizing minority-class misclassification more heavily.
        """
    )

# ── Tuned XGBoost ──────────────────────────────────────────────────────────────
with tab_xgb:
    st.subheader("Tuned XGBoost — Production Model")

    col1, col2 = st.columns(2)
    col1.metric("CV PR-AUC", "0.8522")
    col2.metric("Tuning Method", "RandomizedSearchCV (40 iter, 5-fold)")

    st.markdown("#### Best Hyperparameters")
    st.code(
        """{
    'n_estimators':     500,
    'max_depth':        4,
    'learning_rate':    0.05,
    'subsample':        0.85,
    'colsample_bytree': 0.7,
    'min_child_weight': 3,
    'gamma':            0.2,
    'scale_pos_weight': 1
}""",
        language="python",
    )

    st.markdown("#### Confusion Matrix")
    st.image(
        f"{IMG}/Tuned_XGBoost_Confusion_Matrix.png",
        caption="XGBoost confusion matrix — held-out test set. Balances true positive capture against false positive cost.",
        use_container_width=True,
    )

    st.markdown("#### ROC & Precision-Recall Curves")
    st.image(
        f"{IMG}/Tuned_XGBoost_ROC_Precision_Curves.png",
        caption="XGBoost ROC and PR curves — PR-AUC of 0.8522 substantially exceeds the 0.113 random baseline.",
        use_container_width=True,
    )

# ── Tuned Random Forest ────────────────────────────────────────────────────────
with tab_rf:
    st.subheader("Tuned Random Forest")

    col1, col2 = st.columns(2)
    col1.metric("CV PR-AUC", "0.8277")
    col2.metric("Tuning Method", "RandomizedSearchCV (40 iter, 5-fold)")

    st.markdown("#### Best Hyperparameters")
    st.code(
        """{
    'n_estimators':      600,
    'max_depth':         25,
    'min_samples_leaf':  1,
    'min_samples_split': 2,
    'max_features':      'sqrt',
    'class_weight':      'balanced_subsample'
}""",
        language="python",
    )

    col3, col4 = st.columns(2)
    with col3:
        st.image(
            f"{IMG}/Tuned_Random_Forest_Confusion_Matrix.png",
            caption="Random Forest confusion matrix — slightly higher false negative rate than XGBoost",
            use_container_width=True,
        )
    with col4:
        st.image(
            f"{IMG}/Tuned_Random_Forest_ROC_Precision_Curves.png",
            caption="Random Forest ROC and PR curves — competitive but trails XGBoost on PR-AUC",
            use_container_width=True,
        )

    st.markdown("**Why not selected:** Slightly lower PR-AUC (0.8277 vs 0.8522) and slower inference speed compared to XGBoost.")

    st.divider()
    st.subheader("Baseline Random Forest (Untuned)")
    col5, col6 = st.columns(2)
    with col5:
        st.image(
            f"{IMG}/Random_Forest_Confusion_Matrix.png",
            caption="Baseline Random Forest confusion matrix",
            use_container_width=True,
        )
    with col6:
        st.image(
            f"{IMG}/Random_Forest_ROC_Precision_Cureves.png",
            caption="Baseline Random Forest ROC and PR curves",
            use_container_width=True,
        )

# ── SVM ────────────────────────────────────────────────────────────────────────
with tab_svm:
    st.subheader("SVM Baseline (RBF Kernel)")
    st.markdown(
        """
        - **Config:** `C=1.0`, `class_weight='balanced'`, `probability=True`
        - The RBF kernel captures non-linear class boundaries, but SVM scales poorly with
          large datasets and is outperformed by ensemble methods on tabular data
        """
    )

    col1, col2 = st.columns(2)
    with col1:
        st.image(
            f"{IMG}/SVM_Confusion_Matrix.png",
            caption="SVM confusion matrix — serves as the baseline; lower recall on subscribers than both tuned models",
            use_container_width=True,
        )
    with col2:
        st.image(
            f"{IMG}/SVM_ROC_Precision_Curves.png",
            caption="SVM ROC and PR curves — falls notably below both tuned ensemble methods at high recall thresholds",
            use_container_width=True,
        )

# ── Methodology & Limitations ──────────────────────────────────────────────────
with tab_limits:
    st.subheader("Feature Set (16 Features)")
    st.markdown(
        """
        | Type | Features |
        |------|---------|
        | Numeric (8) | `age`, `campaign`, `consumer_confidence_index`, `euribor_3mo_rate`, `previous_contacted`, `was_previously_contacted`, `positive_campaign_result`, `cluster` |
        | Categorical (8) | `job`, `has_housing_loan`, `has_personal_loan`, `default_status`, `marital_status`, `education_level`, `communication_type`, `last_contact_month` |
        """
    )

    st.subheader("Preprocessing Pipeline")
    st.markdown(
        """
        The model is saved as a `sklearn.pipeline.Pipeline` — calling `.predict()` or
        `.predict_proba()` on raw input automatically applies all transformations:

        - **Numeric columns:** `StandardScaler`
        - **Categorical columns:** `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`

        No manual preprocessing is required when using the saved pipeline.
        """
    )

    st.subheader("Tuning Process")
    st.markdown(
        """
        - `RandomizedSearchCV` with **40 iterations** and **5-fold stratified cross-validation**
        - Optimization target: `average_precision` (equivalent to PR-AUC)
        - Stratified folds ensure each fold maintains the 11.3% positive class ratio
        """
    )

    st.subheader("Known Limitations")
    st.markdown(
        """
        1. **`last_contact_duration_sec` is excluded.** It is a strong predictor but only observable
           *after* the call ends — using it would constitute data leakage for a pre-call scoring system.
        2. **Cluster feature requires pre-computation.** In production, the customer's cluster
           assignment must be derived from the saved clustering pipeline before scoring.
        3. **Temporal generalization.** The dataset covers May 2008 – November 2010. Model performance
           may degrade in macroeconomic environments substantially different from the training period.
        4. **Class imbalance.** Despite weighting, the model will still misclassify some subscribers.
           Threshold tuning (lowering the 0.5 default) can increase recall at the cost of precision.
        """
    )
