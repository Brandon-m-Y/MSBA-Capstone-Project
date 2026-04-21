import streamlit as st


st.set_page_config(
    page_title="Brandon Ytuarte | UCI Bank Marketing",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================== LOGO ======================
# st.logo()

# ====================== DEFINE PAGES ======================


# Create StreamlitPage objects once
home_page = st.Page("1_Home.py", title="Home")
analysis_page = st.Page("2_General_Analysis.py", title="Exploratory Data Analysis")
segmentation_page = st.Page("3_Segmentation_Anlaysis.py", title="Marketing Segmentation")
modeling_page = st.Page("4_Model_Prediction.py", title="Predict New Customers")
models_page = st.Page('5_Model_Assumptions.py', title='About The Models')
data_page = st.Page('6_About_the_data.py', title='About The Data')

# ====================== SIDEBAR NAVIGATION ======================
st.sidebar.title("Navigation Pane:")

st.sidebar.write("Click on a page below to get started!", '\n')

# Custom sidebar links using the same Page objects (this fixes the KeyError)
st.sidebar.page_link(home_page, label="Home")
st.sidebar.page_link(analysis_page, label="EDA")
st.sidebar.page_link(segmentation_page, label="Marketing Segmentation")
st.sidebar.page_link(modeling_page, label="Customer Prediction")
st.sidebar.page_link(models_page, label="About The Models")
st.sidebar.page_link(data_page, label="About The Data")


# ====================== HIDDEN NAVIGATION FOR ROUTING ======================
pg = st.navigation(
    [home_page, analysis_page, segmentation_page, modeling_page, data_page, models_page],
    position="hidden"   # Hides the automatic navigation
)

pg.run()