import streamlit as st

st.header("About the Bank Marketing Data")
st.divider()

st.markdown(
    """
    ### Dataset Overview

    The **Bank Marketing Dataset** originates from the **UCI Machine Learning Repository** and
    captures real outbound calling campaigns run by a **Portuguese banking institution** between
    **May 2008 and November 2010**. The goal of each campaign was to convince clients to subscribe
    to a **term deposit** (a fixed-income savings product).

    | Attribute | Value |
    |-----------|-------|
    | **Source** | UCI Machine Learning Repository |
    | **Donated** | February 13, 2012 |
    | **Authors** | Sérgio Moro, Paulo Cortez, Paulo Rita |
    | **Study Period** | May 2008 – November 2010 |
    | **Raw Records** | 41,188 rows × 21 columns |
    | **Records after cleaning** | 39,803 rows × 22 columns |
    | **Target variable** | `term_deposit_subscribed` (yes / no) |
    | **Class balance** | 88.7% no / 11.3% yes |
    """
)

st.divider()

st.subheader("Feature Definitions")

tab_raw, tab_engineered, tab_dropped = st.tabs([
    "Original Features",
    "Engineered Features",
    "Columns Dropped",
])

with tab_raw:
    st.markdown("##### Client Demographics")
    st.markdown(
        """
        | Column (cleaned name) | Original name | Description |
        |-----------------------|---------------|-------------|
        | `age` | `age` | Client age in years |
        | `job` | `job` | Type of job (11 categories) |
        | `marital_status` | `marital` | Marital status — married / single / divorced |
        | `education_level` | `education` | Highest education level (7 levels) |
        | `default_status` | `default` | Has credit in default? — no / yes / unknown |
        | `has_housing_loan` | `housing` | Has housing loan? — yes / no |
        | `has_personal_loan` | `loan` | Has personal loan? — yes / no |
        """
    )

    st.markdown("##### Campaign Contact")
    st.markdown(
        """
        | Column (cleaned name) | Original name | Description |
        |-----------------------|---------------|-------------|
        | `communication_type` | `contact` | Contact method — telephone / cellular |
        | `last_contact_month` | `month` | Month of last contact |
        | `last_contact_duration_sec` | `duration` | Duration of last call in seconds *(excluded from classification — post-call only)* |
        | `campaign` | `campaign` | Number of contacts during this campaign |
        | `previous_contacted` | `previous` | Number of contacts before this campaign |
        | `previous_campaign_result` | `poutcome` | Outcome of previous campaign *(dropped — 86.4% "nonexistent")* |
        """
    )

    st.markdown("##### Macroeconomic Indicators")
    st.markdown(
        """
        | Column (cleaned name) | Original name | Description |
        |-----------------------|---------------|-------------|
        | `employment_variation_rate` | `emp.var.rate` | Quarterly employment variation rate *(dropped — multicollinear)* |
        | `consumer_price_index` | `cons.price.idx` | Monthly CPI *(dropped — multicollinear)* |
        | `consumer_confidence_index` | `cons.conf.idx` | Monthly consumer confidence index |
        | `euribor_3mo_rate` | `euribor3m` | Euribor 3-month rate — **strongest macro predictor** |
        | `num_employees` | `nr.employed` | Number of bank employees *(dropped — multicollinear)* |
        """
    )

with tab_engineered:
    st.markdown(
        "Three binary indicator columns were created during cleaning before any sparsity-driven drops:"
    )
    st.markdown(
        """
        | New Column | Definition | Positive Rate |
        |------------|------------|:-------------:|
        | `was_previously_contacted` | 1 if the customer was contacted in a prior campaign | 3.7% |
        | `positive_campaign_result` | 1 if the prior campaign resulted in a subscription | 13.6% |
        | `default_status_known` | 1 if default status is explicitly known (not "unknown") | 79.1% |
        """
    )
    st.markdown(
        "Additionally, `education_level` unknowns were **mode-imputed within each job group** "
        "rather than dropped, retaining those rows for modeling."
    )

with tab_dropped:
    st.markdown(
        """
        | Column | Reason for Removal |
        |--------|--------------------|
        | `days_since_prev_campaign_contact` | 96.3% were value 999 ("never contacted") — redundant with `was_previously_contacted` flag |
        | `previous_campaign_result` | 86.4% "nonexistent" — redundant with `positive_campaign_result` flag |
        | `employment_variation_rate` | Highly multicollinear with Euribor and other macro indicators |
        | `num_employees` | Highly multicollinear with employment variation rate |
        | `consumer_price_index` | Highly multicollinear with other macro indicators |
        | `last_contact_duration_sec` | Post-call observation only — excluded from classification to prevent data leakage |
        """
    )
    st.markdown(
        "Additionally, rows where `job`, `marital_status`, `has_housing_loan`, or `has_personal_loan` "
        "were coded as 'unknown' were **dropped** (combined ~1,385 rows, ~3.4% of raw data)."
    )

st.divider()

st.subheader("Citation")
st.markdown(
    """
    **APA:**
    Moro, S., Rita, P., & Cortez, P. (2014). Bank Marketing [Dataset]. UCI Machine Learning Repository.
    https://doi.org/10.24432/C5K306

    **Journal Paper:**
    Moro, S., Cortez, P., & Rita, P. (2014). A data-driven approach to predict the success of bank
    telemarketing. *Decision Support Systems*, 62, 22–31.
    """
)
