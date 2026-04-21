## **Phase 2:** Univariate Analysis – Understand Individual Variables

### Numerical features:

1) What is the age distribution of our customer base? Any outliers or age groups that dominate?

2) How many times are customers typically contacted in this campaign (campaign)? What does the distribution of last_contact_duration_sec look like?

3) How many prior contacts (previous_contacted, was_previously_contacted)? What % of customers were never contacted before?

4) How do the macroeconomic variables (consumer_confidence_index, euribor_3m_rate) vary? Are they constant across the dataset (time-based) or customer-specific?

### Categorical features:

5) What are the top 5 most common job categories and education levels? Any rare categories we should group?

6) What is the balance of marital_status, default_status, has_housing_loan, has_personal_loan?

7) When do we contact people (last_contact_month, day_of_week, communication_type)? Any clear patterns in timing?

### Target balance:

8) What is the overall subscription rate (term_deposit_subscribed mean)? Is the class imbalance severe (common in marketing — often ~10–15%)? 

9) Same question for positive_campaign_result.

10) Visuals to create: histograms/boxplots for numerical, countplots for categorical, value_counts() + bar plots.

__________________________________________________________________________________________________________________

## **Phase 3:** Bivariate & Target Relationship Analysis (Core Marketing Insights)

1) Focus on how features relate to subscription success.

2) Does subscription rate vary significantly by demographics?


3) Age groups vs. subscription rate
Job / education_level / marital_status vs. subscription rate
Loan status (housing/personal/default) vs. subscription rate


4) Does campaign behavior predict success?


5) last_contact_duration_sec vs. subscription (longer calls = higher conversion?)
Number of contacts (campaign) vs. subscription (diminishing returns?)
Previous contact history (was_previously_contacted, previous_contacted) vs. subscription


6) Does contact channel and timing matter?


7) communication_type (telephone vs cellular?) vs. subscription
day_of_week and last_contact_month vs. subscription rate


8) Are there interactions? (e.g., job + education, or age + loan status)
Correlation matrix among numerical features (age, campaign, last_contact_duration_sec, economic indicators). Any strong multicollinearity?

9) Visuals: grouped bar plots (subscription rate by category), boxplots (duration by subscription), correlation heatmap, pairplots.
Key metric to compute repeatedly: subscription rate = df.groupby('feature')['term_deposit_subscribed'].mean()

______________________________________________________________________________________________________________

## **Phase 4:** Multivariate & Segment Exploration (Bridge to Clustering)

1) Can we already see natural customer “types” just by cross-tabulating a few variables?
(e.g., young professionals with no loans vs. older customers with housing loans)

2) How do economic indicators interact with customer features? (e.g., high consumer confidence + certain jobs → higher subscription?)

3) Are there any redundant or unhelpful features for segmentation?


4) Should we keep both economic variables (they may be highly correlated)?

5) Is default_status_known useful or just metadata?

_________________________________________________________________________________________

## **Phase 5:** Preparation for Clustering / Customer Segmentation

### Once EDA is complete, answer these to choose features and methodology:

1) Which feature set should we use for clustering?
Recommended starting set (marketing-relevant):


2) Demographics: age (scaled), job, marital_status, education_level

3) Financial: has_housing_loan, has_personal_loan, default_status

4) Behavioral: previous_contacted, was_previously_contacted, communication_type

5) Campaign context: last_contact_month, day_of_week (optional)

6) → Drop target variables (term_deposit_subscribed, positive_campaign_result) and any pure index columns.

7) Do we need feature engineering first?

8) Bin age into groups?

9) One-hot encode or target-encode high-cardinality categoricals (job, education)?

10) Scale numerical features?

11) Reduce dimensionality (PCA) if using many features?


12) What distance metric and algorithm make sense? (K-Means for interpretability? Hierarchical for dendrogram insight? DBSCAN if noisy clusters?)


13) How will we validate segments? (silhouette score, business interpretability via subscription rate per cluster)