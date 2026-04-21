import streamlit as st

st.header("Digging Deeper With Customer Segmentation")
st.caption(
    "Agglomerative Clustering (Ward linkage, k=3) segments 39,803 customers into three "
    "behaviorally distinct groups — each with a distinct conversion rate and marketing strategy."
)
st.divider()

IMG = "Images/Clustering_analysis_images"

# ── Cluster Summary Table ──────────────────────────────────────────────────────
st.subheader("Cluster Summary")
st.markdown(
    """
    | Cluster | Label | Size | Subscription Rate | Key Driver |
    |---------|-------|:----:|:-----------------:|------------|
    | **0** | Repeat Responders | 13.65% (5,433) | 26.43% | Prior positive campaign result |
    | **1** | Engaged Long-Callers | 20.90% (8,320) | **36.67%** | Long calls + low Euribor |
    | **2** | Cold Prospects | 65.45% (26,050) | 0.00% | High Euribor environment |
    """
)

st.divider()

# ── Tabs for overview vs. detailed breakdowns ──────────────────────────────────
tab_overview, tab_c0, tab_c1, tab_c2, tab_demo = st.tabs([
    "Overview Charts",
    "Cluster 0 — Repeat Responders",
    "Cluster 1 — Engaged Long-Callers",
    "Cluster 2 — Cold Prospects",
    "Demographic Breakdowns",
])

# ── Overview ──────────────────────────────────────────────────────────────────
with tab_overview:
    st.subheader("Subscription Rate by Cluster")
    st.markdown(
        "Clusters 0 and 1 convert at 26–37% while Cluster 2 has **zero conversions**. "
        "A campaign targeting only Clusters 0 and 1 (~34% of the population) would capture "
        "nearly all subscriptions."
    )
    col1, col2 = st.columns(2)
    with col1:
        st.image(f"{IMG}/Percent_Subscribed_by_Cluster.png",
                 caption="Subscription rate (%) by cluster",
                 use_container_width=True)
    with col2:
        st.image(f"{IMG}/Subscribers_by_Cluster.png",
                 caption="Total subscribers concentrated in Clusters 0 and 1",
                 use_container_width=True)

    st.subheader("Euribor Rate — The Primary Separator")
    st.markdown(
        "The Euribor 3-month rate is the **dominant numerical separator** across clusters. "
        "Cluster 2's high rate (~4.63) correlates directly with its zero conversion rate."
    )
    col3, col4 = st.columns(2)
    with col3:
        st.image(f"{IMG}/Mean_Euribor_Rate_by_Cluster.png",
                 caption="Mean Euribor rate by cluster",
                 use_container_width=True)
    with col4:
        st.image(f"{IMG}/Median_Euribor_by_Cluster.png",
                 caption="Median Euribor rate by cluster — confirms the pattern is structural, not outlier-driven",
                 use_container_width=True)

    st.subheader("Call Duration")
    st.markdown(
        "Cluster 1 customers have the **longest median call duration** (~376 sec). "
        "Longer, more engaged conversations are strongly associated with conversion."
    )
    col5, col6 = st.columns(2)
    with col5:
        st.image(f"{IMG}/Median_Last_Contact_by_Cluster.png",
                 caption="Median last contact duration by cluster",
                 use_container_width=True)
    with col6:
        st.image(f"{IMG}/Mean_Age_by_Cluster.png",
                 caption="Mean age is nearly identical across clusters — segmentation is behavioral, not demographic",
                 use_container_width=True)

    st.subheader("Feature Heatmap")
    st.image(f"{IMG}/Numeric_Cat_Means_by_Cluster.png",
             caption="Numeric and categorical feature means by cluster — Euribor rate and campaign contacts dominate the visual contrast",
             use_container_width=True)

# ── Cluster 0 ─────────────────────────────────────────────────────────────────
with tab_c0:
    st.subheader("Cluster 0 — Repeat Responders (13.65% of customers)")

    col1, col2, col3 = st.columns(3)
    col1.metric("Cluster Size", "5,433 (13.65%)")
    col2.metric("Subscription Rate", "26.43%")
    col3.metric("Mean Euribor Rate", "1.50")

    st.markdown(
        """
        **Profile:** Customers who were **previously contacted and succeeded in prior campaigns**.
        All records carry `positive_campaign_result = 1`, explaining the elevated 26.4% subscription rate.
        Euribor rates are low (1.50), placing these interactions in a favorable rate environment.
        Call duration is moderate (mean ~266 sec).

        **Marketing Strategy:** These are **warm leads** — previously engaged and proven responsive.
        Prioritize for retention outreach and cross-sell campaigns. Minimal persuasion budget needed.
        """
    )

# ── Cluster 1 ─────────────────────────────────────────────────────────────────
with tab_c1:
    st.subheader("Cluster 1 — Engaged Long-Callers (20.90% of customers) ⭐ Highest Value")

    col1, col2, col3 = st.columns(3)
    col1.metric("Cluster Size", "8,320 (20.90%)")
    col2.metric("Subscription Rate", "36.67%")
    col3.metric("Mean Call Duration", "376 sec")

    st.markdown(
        """
        **Profile:** The **highest-converting segment** at 36.67%. These customers were never
        previously contacted but responded to longer, more engaged conversations (mean 376 sec —
        the longest of all clusters). Euribor rates are low (1.83), similar to Cluster 0.
        85.8% of this cluster uses **cellular** contact.

        **Marketing Strategy:** Invest in **call quality over call volume**. A single well-executed
        conversation is highly effective. Screen for cellular contact method and low-Euribor periods.
        Do not rely on prior relationship — fresh, quality outreach wins here.
        """
    )

# ── Cluster 2 ─────────────────────────────────────────────────────────────────
with tab_c2:
    st.subheader("Cluster 2 — Cold Prospects (65.45% of customers)")

    col1, col2, col3 = st.columns(3)
    col1.metric("Cluster Size", "26,050 (65.45%)")
    col2.metric("Subscription Rate", "0.00%")
    col3.metric("Mean Euribor Rate", "4.63")

    st.markdown(
        """
        **Profile:** The largest segment with a **zero subscription rate**. These customers were
        never previously contacted and were reached during significantly higher Euribor periods
        (4.63) — suggesting macroeconomic timing, not customer quality, drives their non-conversion.
        Call duration is shortest (mean ~220 sec).

        **Marketing Strategy:** **Deprioritize during high-Euribor periods.** If contacted, frame
        messaging around rate comparisons or fixed-income security. Consider re-activating this
        segment when rates fall — their profile is not inherently worse, just poorly timed.
        """
    )

# ── Demographic Breakdowns ─────────────────────────────────────────────────────
with tab_demo:
    st.subheader("Demographics Across Clusters")
    st.markdown(
        "Job, education, and marital status distributions are broadly similar across all three "
        "clusters — confirming that **segmentation is driven by behavioral and macroeconomic "
        "features, not demographics**."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.image(f"{IMG}/Job_Type_by_Cluster.png",
                 caption="Job type distribution by cluster",
                 use_container_width=True)
    with col2:
        st.image(f"{IMG}/Education_by_Cluster.png",
                 caption="Education level distribution by cluster",
                 use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.image(f"{IMG}/Marital_Status_by_Cluster.png",
                 caption="Marital status by cluster",
                 use_container_width=True)
    with col4:
        st.image(f"{IMG}/Communication_Type_by_Cluster.png",
                 caption="Communication type by cluster — cellular dominates Clusters 0 and 1",
                 use_container_width=True)

    st.subheader("Loan & Default Status")
    col5, col6, col7 = st.columns(3)
    with col5:
        st.image(f"{IMG}/Housing_Loans_by_Cluster.png",
                 caption="Housing loan ownership by cluster",
                 use_container_width=True)
    with col6:
        st.image(f"{IMG}/Personal_Loans_by_Cluster.png",
                 caption="Personal loan ownership by cluster",
                 use_container_width=True)
    with col7:
        st.image(f"{IMG}/Defaults_by_Cluster.png",
                 caption="Credit default rates by cluster",
                 use_container_width=True)
