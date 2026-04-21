import streamlit as st

st.header("Exploratory Data Analysis")
st.caption("Key patterns in the UCI Bank Marketing dataset — 39,803 records after cleaning.")
st.divider()

IMG = "Images/General_data_analysis_images"

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Subscription Overview",
    "Age & Demographics",
    "Job & Education",
    "Contact Timing",
    "Macroeconomic Context",
    "Correlations & Cross-Features",
])

# ── Tab 1: Subscription Overview ──────────────────────────────────────────────
with tab1:
    st.subheader("Overall Subscription Rate")
    st.markdown(
        "Only **~11.3%** of contacted customers subscribed to the term deposit. "
        "This severe class imbalance means standard accuracy metrics are misleading — "
        "PR-AUC and minority-class F1 are prioritized throughout modeling."
    )
    st.image(f"{IMG}/Overall_Subscription_Rate.png", use_container_width=True)

# ── Tab 2: Age & Demographics ──────────────────────────────────────────────────
with tab2:
    st.subheader("Age Distribution")
    st.markdown(
        "The customer base skews toward working-age adults (30–50), with a secondary peak "
        "among retirees. Age alone is a weak predictor but interacts meaningfully with "
        "economic context."
    )
    col1, col2 = st.columns(2)
    with col1:
        st.image(f"{IMG}/Age_Distribution.png", caption="Overall age distribution", use_container_width=True)
    with col2:
        st.image(f"{IMG}/Subscriber_Age_Distribution.png", caption="Age distribution for subscribers vs. non-subscribers", use_container_width=True)

    st.subheader("Marital Status")
    st.markdown(
        "Married customers dominate the sample, but **single customers show the highest "
        "per-capita conversion rate** — possibly reflecting greater financial independence "
        "or product appeal."
    )
    col3, col4 = st.columns(2)
    with col3:
        st.image(f"{IMG}/Mariage_Status_Distributions.png", caption="Marital status distribution", use_container_width=True)
    with col4:
        st.image(f"{IMG}/Marital_Staues_by_Education.png", caption="Marital status broken down by education level", use_container_width=True)

# ── Tab 3: Job & Education ──────────────────────────────────────────────────────
with tab3:
    st.subheader("Job Type")
    st.markdown(
        "Admin, blue-collar, and technician roles dominate the contact list, but "
        "**retired and student categories convert at disproportionately high rates**, "
        "signaling priority targeting opportunities."
    )
    col1, col2 = st.columns(2)
    with col1:
        st.image(f"{IMG}/Overall_Job_Distribution.png", caption="Overall job distribution in the dataset", use_container_width=True)
    with col2:
        st.image(f"{IMG}/Percent_Subscribers_by_job.png", caption="Subscription rate by job category", use_container_width=True)

    st.image(f"{IMG}/Subscription_Rate_Lift_Job_Category.png",
             caption="Subscription rate lift by job — segments above 1.0x outperform the campaign average",
             use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.image(f"{IMG}/Subscribers_by_job_type.png", caption="Subscriber count by job type", use_container_width=True)
    with col4:
        st.image(f"{IMG}/NonSubscribers_by_job_type.png", caption="Non-subscriber count by job type", use_container_width=True)

    st.divider()
    st.subheader("Education Level")
    st.markdown(
        "Education level drives a meaningful subscription rate gradient. Illiterate customers "
        "convert poorly, while **university graduates respond well**, likely reflecting greater "
        "financial product familiarity."
    )
    col5, col6 = st.columns(2)
    with col5:
        st.image(f"{IMG}/Overall_Education_Distribution.png", caption="Overall education distribution", use_container_width=True)
    with col6:
        st.image(f"{IMG}/Percent_Subscribers_by_Education.png", caption="Subscription rate by education level", use_container_width=True)

    st.image(f"{IMG}/Subscription_Rate_Lift_Education.png",
             caption="Subscription rate lift by education — segments above 1.0x outperform the campaign average",
             use_container_width=True)

    col7, col8 = st.columns(2)
    with col7:
        st.image(f"{IMG}/Subscribers_by_Education.png", caption="Subscriber count by education level", use_container_width=True)
    with col8:
        st.image(f"{IMG}/Job_Types_by_Education.png", caption="Job type distribution across education levels", use_container_width=True)

# ── Tab 4: Contact Timing ──────────────────────────────────────────────────────
with tab4:
    st.subheader("Monthly Contact Volume vs. Conversion")
    st.markdown(
        "Contact volume peaks in **May**, but conversion rates peak in **March, September, "
        "October, and December** — spring 'push' campaigns are reaching many but converting few."
    )
    col1, col2 = st.columns(2)
    with col1:
        st.image(f"{IMG}/Contact_by_Month.png", caption="Total contacts by month", use_container_width=True)
    with col2:
        st.image(f"{IMG}/Subscription_Rates_by_Month.png", caption="Subscription rate by month", use_container_width=True)

    st.subheader("Communication Channel")
    st.markdown(
        "**Cellular contact achieves significantly higher subscription rates than telephone** — "
        "a strong operational signal for channel prioritization in future campaigns."
    )
    st.image(f"{IMG}/Subscription_Rates_by_Communication_type.png",
             caption="Subscription rates by communication channel",
             use_container_width=True)

    st.subheader("Call Duration & Contact Frequency")
    st.markdown(
        "Calls lasting **300+ seconds correlate strongly with subscription** — though call "
        "duration is only observable *after* a call and cannot be used as a pre-call feature. "
        "Subscribers also receive *fewer* repeat contacts on average, suggesting quality "
        "engagement in early calls matters more than persistence."
    )
    col3, col4 = st.columns(2)
    with col3:
        st.image(f"{IMG}/Subscriber_Last_Contact_Duration.png", caption="Last contact duration for subscribers vs. non-subscribers", use_container_width=True)
    with col4:
        st.image(f"{IMG}/Subscriber_Contact_Frequency.png", caption="Number of campaign contacts for subscribers vs. non-subscribers", use_container_width=True)

    col5, col6 = st.columns(2)
    with col5:
        st.image(f"{IMG}/Last_Contact_Times_Distribution.png", caption="Distribution of last contact duration (all customers)", use_container_width=True)
    with col6:
        st.image(f"{IMG}/Previous_Contact_Distribution.png", caption="Distribution of prior campaign contacts", use_container_width=True)

    st.subheader("Mean Contact Rates")
    st.image(f"{IMG}/Mean_Contact_Rates.png", caption="Mean contact rates across the dataset", use_container_width=True)
    st.image(f"{IMG}/Subscribers_Mean_Contact_Rate.png", caption="Mean contact rate for subscribers", use_container_width=True)

# ── Tab 5: Macroeconomic Context ───────────────────────────────────────────────
with tab5:
    st.subheader("Euribor 3-Month Rate")
    st.markdown(
        "**Euribor rate is the single strongest macro-level predictor.** Term deposits are most "
        "attractive when rates are low (~1.5–2%), as customers seek stable yield alternatives. "
        "Campaigns during high-rate periods (Euribor > 4%) are nearly ineffective."
    )
    col1, col2 = st.columns(2)
    with col1:
        st.image(f"{IMG}/Mean_Euribor_for_Subscribers.png", caption="Mean Euribor rate — subscribers vs. non-subscribers", use_container_width=True)
    with col2:
        st.image(f"{IMG}/Mean_Euribor_by_Month.png", caption="Mean Euribor rate by month across the study period", use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.image(f"{IMG}/Subscribers_Euribor_Education.png", caption="Euribor rate for subscribers broken down by education", use_container_width=True)
    with col4:
        st.image(f"{IMG}/Mean_Euribor_Rate_by_Cluster.png" if False else f"{IMG}/Subuscribers_Age_Distribution.png",
                 caption="Subscriber age distribution (detailed)", use_container_width=True)

# ── Tab 6: Correlations & Cross-Features ──────────────────────────────────────
with tab6:
    st.subheader("Numerical Correlation Heatmap")
    st.markdown(
        "Strong multicollinearity between employment variation rate, number of employees, "
        "and consumer price index justifies removing those columns before modeling."
    )
    st.image(f"{IMG}/Heatmap_for_Numerical_Correlation.png",
             caption="Pearson correlation heatmap for all numeric features",
             use_container_width=True)

    st.subheader("Housing & Personal Loans by Education")
    col1, col2 = st.columns(2)
    with col1:
        st.image(f"{IMG}/Home_Loans_by_Education.png", caption="Housing loan ownership by education level", use_container_width=True)
    with col2:
        st.image(f"{IMG}/Personal_Loan_by_Education.png", caption="Personal loan ownership by education level", use_container_width=True)
