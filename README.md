# Personalized Credit Card / Term Deposit Offer Acceptance Prediction

Predict whether a client will accept a direct marketing offer (term deposit as proxy for credit card/personal loan) using the UCI Bank Marketing dataset.

Dataset:  
- UCI Bank Marketing (bank-additional-full.csv)  
- 41,188 rows, 21 columns  
- Target: `y` (yes/no acceptance) – highly imbalanced (~11.3% yes)

Key Features:
- Realistic modeling: Excluded call duration (post-call leakage)
- Best model: **XGBoost** (tuned) – AUC **0.8164** on test set
- Strongest drivers: Previous campaign success (`poutcome_success`), euribor3m (negative), pdays, contact_cellular

Tech Stack:
- Python
- Pandas & NumPy
- Matplotlib & Seaborn (EDA & visualization)
- Scikit-learn (preprocessing, Logistic Regression, evaluation)
- XGBoost (gradient boosting)
- Git & GitHub

Project Highlights:
- Thorough EDA: Acceptance by contact method, previous outcome, month, job, age group, economic indicators
- Feature engineering: Numeric target, age groups, duration in minutes (for visualization only)
- Models compared: Logistic Regression (~0.80), Random Forest (~0.81), CatBoost (~0.81), XGBoost (**0.8164**)
- Business insights: Prioritize callbacks to previous successes, prefer cellular contact, time campaigns for low euribor3m periods

Results:
- Best AUC: **0.8164** (XGBoost)
- Top feature: `poutcome_success` – massive uplift for past responders

How to Run:
1. Clone repo
2. Install dependencies: `pip install -r requirements.txt` (or just pandas, matplotlib, seaborn, scikit-learn, xgboost)
3. Run the notebook: `jupyter notebook bank_marketing_project.ipynb`

Dataset Source:  
https://archive.ics.uci.edu/dataset/222/bank+marketing

Screenshots (included in repo):
- Feature importance plot
- Acceptance by previous outcome
- Acceptance by month of contact
- Offer acceptance by jobtype

Feel free to explore the notebook for full EDA, code, and results!
