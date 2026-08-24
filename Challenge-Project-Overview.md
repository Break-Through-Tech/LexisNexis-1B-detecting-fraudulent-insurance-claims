# Detecting Fraudulent Insurance Claims: A Risk-Scoring Model for Faster, Fairer Claims Triage

**Company / Org:** LexisNexis Risk Solutions Group  
**Challenge Advisor:** Stephanie Le, ledaquynhnhi@gmail.com  
**AI Studio Coach:** Ananya Devarakonda, ananya.devarakonda@breakthroughtech.org    
**Program:** Break Through Tech AI Studio - Fall 2026  

---

## 🏢 About LexisNexis Risk Solutions Group
LexisNexis Risk Solutions Group is a global leader in providing data, analytics, and technology solutions to help organizations manage risk and improve decision-making. The team objectives center on leveraging predictive modeling to enhance operational efficiency within the insurance sector, specifically by optimizing claims processing workflows.

---

## 🎯 The Challenge
### Project Summary
In this project, you will use structured historical auto insurance claims data (policy details, claimant demographics, and incident characteristics) and supervised classification techniques (like logistic regression, random forest, gradient boosting) to build a model that scores claims by likelihood of fraud. This will help our company address the challenge of prioritizing limited fraud-investigation resources toward the claims most likely to be fraudulent, reducing losses from fraudulent payouts while minimizing review delays for legitimate claimants

### Success Criteria
Given fraud is a rare-class problem, accuracy alone is misleading. Use precision, recall, F1, and PR-AUC on the minority (fraud) class, plus a business-framed metric like "% of fraud cases captured if investigators review only the top 10–20% highest-scored claims." A successful outcome by December: a working, interpretable model that clearly beats a naive baseline on recall/precision trade-off, with a short explanation of which features drive risk.

### Stretch Goals
cost-sensitive learning (weighting false negatives more heavily, since missed fraud is costlier than false alarms), a lightweight Streamlit demo for exploring flagged claims, stacked/ensemble models, unsupervised anomaly detection as a complementary check, enriching with a second public dataset.

### Project Milestones
Use these milestones to guide your work. Your team will create a GitHub Projects board to track tasks within each milestone.

| Month | Milestone | Key Activities |
|---|---|---|
| September | Foundations: EDA & Baseline Model | Data exploration & cleaning; EDA on fraud vs. non-fraud patterns; review the provided data dictionary; audit the feature list for potential leakage (see Dataset section below) and finalize which columns are safe to use; establish a baseline model (logistic regression) and baseline metrics using a hold-out (not K-Fold) split. |
| October | Feature Engineering & Imbalance Handling | Engineering & Imbalance Handling	Feature engineering; address class imbalance (class weighting, SMOTE); train/tune 2–3 model types (random forest, gradient boosting) and compare, continuing to validate on the same hold-out/temporal split established in September. |
| November | Finalization, Interpretability & Model Card | Finalize best model; evaluate with business-relevant metrics; add interpretability (SHAP or feature importance); write a model card summarizing the model's purpose, performance, key risk drivers, and known limitations; polish GitHub repo, write-up, and final presentation. |

> **Note for the team:** Please create a GitHub Projects board in this repository to break these milestones into weekly tasks. Go to the **Projects** tab → **New project** → Choose **Board** → Add columns for each month.

---

## 📊 Dataset
**Name and Source:** Vehicle Insurance Claim Fraud Detection (Kaggle): 
**Format:** CSV  
**Size:** under 1gb  
**Location:** https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection

### Key Details
- Structured tabular data on auto insurance claims from 1994–1996, covering policy details, claimant demographics, and incident characteristics, with a binary fraud label (FraudFound_P). No missing values, but the fraud class is rare (~6% of claims), so class imbalance needs to be addressed during modeling.
- Feature leakage audit needed: a few fields (eg: PoliceReportFiled, WitnessPresent, NumberOfSuppliments, AddressChange_Claim) may reflect information recorded during or after a fraud investigation rather than at the time the claim was filed. Before modeling, the team should review each feature's timing and drop or flag anything that would leak future information into the model.
- Data dictionary: Kaggle's page does not include a full data dictionary. I will prepare and add one to the /data folder before kickoff, covering each column's name, type, and definition.

---

## 🛠️ Suggested Approach

**ML Problem Type:** Classification  

**Recommended Libraries:**
- pandas, numpy — data handling
- scikit-learn — modeling (logistic regression, random forest, gradient boosting)
- imbalanced-learn — class imbalance handling (SMOTE, class weighting)
- SHAP — model interpretability
- matplotlib / seaborn — EDA and visualization

**Evaluation Metrics:**
- Precision, Recall, F1, and PR-AUC on the fraud (minority) class
- Business-framed metric: % of fraud captured if investigators review only the top 10–20% highest-scored claims
- Validation approach: use a hold-out or temporal split rather than standard K-Fold cross-validation, since the data spans multiple years — this avoids leaking information across time and better reflects how the model would perform on future claims.
  
---

## 📚 Resources to Get Started

The following resources will help your team understand the problem space and potential technical approaches for this project:

**Background Reading:**
- [Figshare: fraud_oracle.csv dataset background](https://figshare.com/articles/dataset/fraud_oracle_csv/24994233?file=44033394) — background on the source study behind this dataset

**Technical Tutorials:**
- [imbalanced-learn documentation](https://imbalanced-learn.org/stable/) — for handling class imbalance (SMOTE, class weighting)
- [SHAP documentation](https://shap.readthedocs.io/) — for the November interpretability/model card work

**Code Examples:**
- [MLOps Basic Open-Source Tool Series: EDA + Deepchecks + Random Forest on this dataset](https://www.nb-data.com/p/mlops-basic-open-source-tool-series-882)
- [Kaggle notebook: Insurance Fraud Detection Using 12 Models](https://www.kaggle.com/code/niteshyadav3103/insurance-fraud-detection-using-12-models)
- [Kaggle notebook: Insurance Fraud Claims Detection](https://www.kaggle.com/code/buntyshah/insurance-fraud-claims-detection)

**Other:**
- [Model Cards for Model Reporting (Google, foundational paper on model card format)](https://arxiv.org/abs/1810.03993) — useful reference for structuring your November model card deliverable

*Feel free to explore beyond these, and share anything interesting you find with me!*

---

## 🤝 How We'll Work Together

**Official check-ins:** During our biweekly 45-minute AI Studio Lab Section meeting block (2nd and 4th week of every month)

 **Other ways to reach out to me with questions:** 
* Your team's channel within Break Through Tech’s Discord space
* Email: please copy your teammates and AI Studio Coach
* Note: I will aim to respond within 48 hours. Please reach out to your AI Studio Coach with urgent questions.

**Recommended free coding / collaboration tools**
- Google Colab (free tier)
- GitHub (repo + Projects board)

---

## 🚀 Getting Started

1. **Review this overview document** and note any questions for our first meeting
2. **Begin reviewing the dataset** using the link above
3. **Read the GitHub Projects documentation** [here](https://docs.github.com/en/issues/planning-and-tracking-with-projects/learning-about-projects/about-projects)

I’m excited to work with you!

---

## ❓ Questions?

Please bring any questions to our first meeting during the week of August 24th (Break Through Tech’s Bridge to Studio - Session C). 
