# Churn Prediction for StreamWorks Media

**Tools:** Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, SciPy  
**Domain:** Subscription / Streaming Media Analytics  
**Dataset:** StreamWorks Media subscriber data

---

## Project Overview

StreamWorks Media is a UK-based video streaming platform facing rising customer churn amid intensifying competition from global players. This project analyses subscriber behaviour to understand churn drivers and builds predictive models to support early intervention by the retention team.

The analysis combines statistical hypothesis testing with two predictive modelling approaches - logistic regression for churn classification and linear regression for engagement prediction.

---

## Business Questions Answered

- Do users who receive promotions churn less?
- Does watch time impact churn likelihood?
- Are mobile-dominant users more likely to cancel?
- What are the top predictors of churn?
- Which customer segments should the retention team prioritise?
- What factors affect user watch time or tenure?

---

## Repository Structure

```
├── Churn Prediction for StreamWorks Media.ipynb   # Full analysis notebook

├── streamworks_user_data.csv                       # Subscriber dataset

└── README.md
```

---

## Key Steps

**1. Data Exploration**
- Reviewed structure, missing values, and distributions using `.info()`, `.describe()`, `.isnull().sum()`
- Built a correlation matrix and heatmap across numeric variables

**2. Data Cleaning & Preparation**
- Converted `signup_date` and `last_active_date` to datetime
- Engineered `tenure_days` and `is_loyal` (tenure > 180 days)
- Handled missing values across numeric and categorical columns

**3. Feature Engineering**
- `watch_per_fee_ratio` — engagement relative to subscription cost
- Label encoding for `gender`, binary encoding for `received_promotions`
- One-hot encoding for `subscription_type` and `country`
- Age binned into groups for segment-level analysis

**4. Statistical Analysis & Hypothesis Testing**
- Chi-square tests for churn association with gender, promotions, and referrals
- Independent t-test comparing watch hours between churned and retained users

**5. Predictive Modelling - Logistic Regression**
- Classification model to predict churn, using `class_weight='balanced'` and a custom 0.3 probability threshold
- Evaluated via confusion matrix, precision/recall/F1, and ROC-AUC

**6. Predictive Modelling - Linear Regression**
- Predicted `average_watch_hours` from user features
- Evaluated via R² and RMSE, with coefficient interpretation

---

## Key Findings

- **No statistically significant predictors of churn** were found among gender (p = 0.107), promotions received (p = 0.115), or friend referral (p = 0.439). A t-test also found no significant watch-time difference between churned and retained users (p = 0.883)
- **The logistic regression model performed poorly** (ROC-AUC = 0.527) - only marginally better than random chance, reinforcing that the available features lack strong churn signal
- **Monthly fee is the strongest driver of watch hours** - the linear regression model performed well (R² = 0.894, RMSE = 7.31 hrs), with `monthly_fee` showing by far the largest coefficient (3.28)
- **Complaints and mobile usage have smaller, directionally sensible effects** on engagement levels

---

## Recommendations

- **Do not deploy the current churn classifier** - its near-random AUC means it would not meaningfully outperform random targeting
- **Invest in richer behavioural data** - session-level activity, support interactions, and billing history would likely provide stronger churn signal
- **Use subscription tier as an engagement lever** - higher-paying tiers correlate strongly with watch hours, suggesting upsell potential
- **Treat statistical non-significance as a genuine finding** - with this feature set, churn does not appear strongly driven by demographics or aggregate usage alone

---

## Data Issues & Limitations

The test set showed class imbalance (234 retained vs 65 churned), which affects classifier reliability. The absence of significant predictors suggests churn at StreamWorks may be driven by factors outside this dataset, such as pricing perception, competitor offers, or content catalogue changes, that would need to be captured for a more effective model.

---

## How to Run

1. Clone this repository
2. Open `Churn Prediction for StreamWorks Media.ipynb` in Jupyter Notebook or JupyterLab
3. Run all cells top to bottom

**Requirements:** `pandas` `numpy` `matplotlib` `seaborn` `scikit-learn` `scipy`

---

*Dataset is fictional and used for educational purposes.*
