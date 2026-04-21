import streamlit as st

st.header("UCI Bank Marketing — Customer Segmentation & Subscription Prediction")
st.caption("Brandon Ytuarte | BMY Analytics | BUSA-695 Capstone")
st.divider()

st.markdown(
    """
    Portuguese banks run telemarketing campaigns to sell **term deposit subscriptions** — a
    critical revenue product with a natural conversion rate of roughly **11%**. Calling every
    customer is expensive and drives churn. This project builds a production-ready ML system
    to answer two strategic questions:

    1. **Who is most likely to subscribe?** A tuned XGBoost classifier predicts individual
       subscription probability so campaign resources can be concentrated on high-yield prospects.
    2. **What kinds of customers exist?** Agglomerative clustering segments the customer base
       into three behaviorally distinct groups, each with different conversion rates, call-duration
       profiles, and macroeconomic sensitivity.
    """
)

st.divider()

# Key metrics
st.subheader("Project at a Glance")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Raw Records", "41,188")
c2.metric("Records After Cleaning", "39,803")
c3.metric("Positive Class Rate", "11.3%")
c4.metric("XGBoost PR-AUC", "0.8522")

st.divider()

# Cluster summary
st.subheader("Customer Segments Discovered")
st.markdown(
    """
    | Cluster | Label | Size | Subscription Rate | Key Driver |
    |---------|-------|:----:|:-----------------:|------------|
    | 0 | Repeat Responders | 13.65% | 26.43% | Prior positive campaign result |
    | 1 | Engaged Long-Callers | 20.90% | **36.67%** | Long calls + low Euribor |
    | 2 | Cold Prospects | 65.45% | 0.00% | High Euribor environment |
    """
)

st.divider()

# Navigation guide
st.subheader("How to Use This App")
st.markdown(
    """
    Use the **Navigation Pane** on the left to move between pages:

    | Page | What You Will Find |
    |------|--------------------|
    | **EDA** | Exploratory data analysis — distributions, subscription rates by segment, macro trends |
    | **Marketing Segmentation** | Cluster profiles, conversion rates, and strategy recommendations |
    | **Customer Prediction** | Enter a customer's attributes and get a real-time subscription probability |
    | **About The Models** | Model comparison, hyperparameter tuning details, confusion matrices, and PR curves |
    | **About The Data** | Dataset description, feature definitions, and citation |
    """
)

st.divider()
st.caption(
    "Data: UCI Machine Learning Repository — Moro, Cortez & Rita (2014). "
    "Period: May 2008 – November 2010."
)
