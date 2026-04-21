import streamlit as st
import pandas as pd
import joblib

st.header('Predict if a New Customer Will Subscribe')
st.divider()

# ================= LOAD MODEL =================
model = joblib.load('Models\\tuned_xgboost_model.pkl')

# Model is a pipeline that includes encoder and scaler.

# ================= INPUTS =================
age = st.slider('age', min_value=18, max_value=80, value=30)

job = st.selectbox('job', [
    'housemaid', 'services', 'admin.', 'blue-collar', 'technician',
    'retired', 'management', 'unemployed', 'self-employed',
    'entrepreneur', 'student'
])

campaign = st.number_input('campaign', min_value=0, max_value=45, value=1)

housing_loan = st.selectbox('has_housing_loan', ['no', 'yes'])
personal_loan = st.selectbox('has_personal_loan', ['no', 'yes'])
default_status = st.selectbox('default_status', ['no', 'unknown', 'yes'])

cc_index = st.slider('consumer_confidence_index', min_value=-60.0, value=0.0)
euribor_rate = st.slider('euribor_3mo_rate', min_value=0.0, value=7.0)

marital_status = st.selectbox('marital_status', ['married', 'single', 'divorced'])

education = st.selectbox('education_level', [
    'basic.4y', 'high.school', 'basic.6y', 'basic.9y',
    'professional.course', 'university.degree', 'illiterate'
])

previously_contacted = st.slider('previous_contacted', min_value=0, max_value=8, value=1)

communication_type = st.selectbox('communication_type', ['telephone', 'cellular'])

last_contact_month = st.selectbox('last_contact_month', [
    'may', 'jun', 'jul', 'aug', 'oct', 'nov', 'dec', 'mar', 'apr', 'sep'
])

was_previously_contacted = st.selectbox('was_previously_contacted', [0, 1])
positive_campaign_result = st.selectbox('positive_campaign_result', [0, 1])

cluster = st.selectbox('cluster', [0, 1, 2])

# ================= BUILD INPUT DATA =================
input_df = pd.DataFrame({
    'age': [age],
    'job': [job],
    'campaign': [campaign],
    'has_housing_loan': [housing_loan],
    'has_personal_loan': [personal_loan],
    'default_status': [default_status],
    'consumer_confidence_index': [cc_index],
    'euribor_3mo_rate': [euribor_rate],
    'marital_status': [marital_status],
    'education_level': [education],
    'previous_contacted': [previously_contacted],
    'communication_type': [communication_type],
    'last_contact_month': [last_contact_month],
    'was_previously_contacted': [was_previously_contacted],
    'positive_campaign_result': [positive_campaign_result], 
    'cluster': [cluster]
})

# ================= ENCODING =================

# Encoding and Scaling is Including in the Model as a Pipline.

# ================= PREDICTION =================
if st.button("Predict Subscription"):

    try:
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        if prediction == 1:
            st.success(f"✅ Customer is likely to subscribe ({probability:.2%} confidence)")
        else:
            st.error(f"❌ Customer is unlikely to subscribe ({probability:.2%} confidence)")

    except Exception as e:
        st.error("⚠️ Prediction failed. Check feature alignment with model.")
        st.text(str(e))
        