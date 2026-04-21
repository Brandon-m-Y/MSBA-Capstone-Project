import streamlit as st

st.header("About The Bank Marketing Data")
st.divider()


st.markdown("""  
### About the Dataset

The **Bank Marketing Dataset** is a classic machine learning dataset originating from the **UCI Machine Learning Repository**.

It contains data from **direct marketing campaigns** (primarily phone calls) conducted by a **Portuguese banking institution** between May 2008 and November 2010. 
The goal of these campaigns was to convince clients to subscribe to a **long-term deposit** (term deposit).

#### Key Information
- **Source**: UCI Machine Learning Repository  
- **Donated**: February 13, 2012  
- **Creators**: Sérgio Moro, Paulo Cortez, and Paulo Rita (2014)  
- **Paper**: "A Data-Driven Approach to Predict the Success of Bank Telemarketing" published in *Decision Support Systems*

#### Dataset Details
- **Total Instances**: 41,188 (full version - `bank-additional-full.csv`)  
- **Features**: 20 input variables + 1 target variable  
- **Target Variable**: `y` (or `term_deposit_subscribed`) — binary classification  
  - **Yes**: Client subscribed to a term deposit  
  - **No**: Client did not subscribe  

The dataset includes three main types of features:
- **Client demographic data** — age, job, marital status, education, housing loan, personal loan, etc.  
- **Campaign-related data** — contact type, last contact month/day, number of contacts, outcome of previous campaigns, etc.  
- **Macroeconomic indicators** — employment variation rate, consumer price index, consumer confidence index, Euribor 3-month rate, and number of employees.

This dataset is widely used in the machine learning community for **binary classification** tasks, particularly to evaluate models on **imbalanced data** (only about 11–12% of clients subscribed).

---
**Citation (APA)**:  
Moro, S., Rita, P., & Cortez, P. (2014). Bank Marketing [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306
""")

