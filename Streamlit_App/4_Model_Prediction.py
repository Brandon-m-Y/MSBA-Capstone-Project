import streamlit as st
import pandas as pd
import joblib

st.header("Predict if a New Customer Will Subscribe")
st.caption(
    "Enter a customer's attributes below and click **Predict Subscription** to get a "
    "real-time probability score from the tuned XGBoost model."
)
st.divider()

# ── Load Model ─────────────────────────────────────────────────────────────────
model = joblib.load("Models/tuned_xgboost_model.pkl")

# ── Input Form ─────────────────────────────────────────────────────────────────
st.subheader("Customer Demographics")
col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age", min_value=18, max_value=80, value=38)

with col2:
    job = st.selectbox("Job", [
        "admin.", "technician", "blue-collar", "management", "services",
        "retired", "self-employed", "entrepreneur", "housemaid",
        "unemployed", "student",
    ])

with col3:
    marital_status = st.selectbox("Marital Status", ["married", "single", "divorced"])

col4, col5, col6 = st.columns(3)
with col4:
    education = st.selectbox("Education Level", [
        "university.degree", "high.school", "professional.course",
        "basic.9y", "basic.6y", "basic.4y", "illiterate",
    ])
with col5:
    default_status = st.selectbox("Default Status", ["no", "unknown", "yes"])
with col6:
    housing_loan = st.selectbox("Housing Loan", ["no", "yes"])

col7, col8 = st.columns([1, 2])
with col7:
    personal_loan = st.selectbox("Personal Loan", ["no", "yes"])

st.divider()
st.subheader("Campaign & Contact Details")

col9, col10, col11 = st.columns(3)
with col9:
    communication_type = st.selectbox("Communication Type", ["cellular", "telephone"])
with col10:
    last_contact_month = st.selectbox("Last Contact Month", [
        "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
    ])
with col11:
    campaign = st.number_input(
        "Contacts This Campaign", min_value=1, max_value=45, value=2,
        help="Number of times the customer was contacted during this campaign."
    )

col12, col13 = st.columns(2)
with col12:
    was_previously_contacted = st.selectbox(
        "Previously Contacted?",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No",
        help="Was this customer contacted in a prior campaign?"
    )
with col13:
    positive_campaign_result = st.selectbox(
        "Prior Campaign Successful?",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No",
        help="Did a prior campaign result in a subscription for this customer?"
    )

previous_contacted = st.slider(
    "Number of Prior Campaign Contacts", min_value=0, max_value=8, value=0
)

st.divider()
st.subheader("Macroeconomic Indicators")

col14, col15 = st.columns(2)
with col14:
    euribor_rate = st.slider(
        "Euribor 3-Month Rate",
        min_value=0.0, max_value=8.0, value=1.5, step=0.01,
        help="Lower values (~1.5–2.0) are associated with much higher conversion rates."
    )
with col15:
    cc_index = st.slider(
        "Consumer Confidence Index",
        min_value=-60.0, max_value=0.0, value=-41.0, step=0.1,
    )

st.divider()
st.subheader("Customer Segment")
cluster = st.selectbox(
    "Cluster Assignment",
    [0, 1, 2],
    format_func=lambda x: {
        0: "0 — Repeat Responders (prior positive contact, low Euribor)",
        1: "1 — Engaged Long-Callers (first contact, long calls, low Euribor)",
        2: "2 — Cold Prospects (first contact, high Euribor)",
    }[x],
    help="Assign the customer to the most appropriate segment based on prior contact history and current Euribor environment."
)

# ── Build Input DataFrame ──────────────────────────────────────────────────────
input_df = pd.DataFrame({
    "age": [age],
    "job": [job],
    "campaign": [campaign],
    "has_housing_loan": [housing_loan],
    "has_personal_loan": [personal_loan],
    "default_status": [default_status],
    "consumer_confidence_index": [cc_index],
    "euribor_3mo_rate": [euribor_rate],
    "marital_status": [marital_status],
    "education_level": [education],
    "previous_contacted": [previous_contacted],
    "communication_type": [communication_type],
    "last_contact_month": [last_contact_month],
    "was_previously_contacted": [was_previously_contacted],
    "positive_campaign_result": [positive_campaign_result],
    "cluster": [cluster],
})

# ── Predict ────────────────────────────────────────────────────────────────────
st.divider()
if st.button("Predict Subscription", type="primary"):
    try:
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        if prediction == 1:
            st.success(f"This customer is **likely to subscribe** — {probability:.2%} confidence.")
        else:
            st.error(f"This customer is **unlikely to subscribe** — {probability:.2%} confidence.")

        with st.expander("View input summary"):
            st.dataframe(input_df.T.rename(columns={0: "Value"}))

    except Exception as e:
        st.error("Prediction failed. Check that all inputs are valid.")
        st.text(str(e))
