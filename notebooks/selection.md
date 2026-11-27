# Model selection

## Selected model: Random Forest

We originally selected Random Forest using the 1995 validation results. Our main goal was to catch the most fraud while reviewing only the highest-scoring 20% of claims. We used Average Precision (AP) to break a tie.

Random Forest caught 151 of 301 fraud cases, compared with 147 for the coverage + fault lookup rule. This small improvement supported the original choice, although the lookup rule had higher AP.

## Model settings and data

- Random Forest: 200 trees, maximum depth 6, minimum 10 samples per leaf.
- Class weights: none. No SMOTE was used for the selected model.
- Random seed: 42.
- Model comparison: train on 1994 and validate on 1995, without K-Fold.
- Final evaluation: refit on 1994–1995 and evaluate on 1996.
- Review rule: rank scores from highest to lowest and review 20%, rounding up. Break tied scores using the same fixed pandas shuffle with seed 42.

We compared logistic regression, Random Forest and gradient boosting across 18 combinations of settings and imbalance methods. The original logistic regression and lookup rule were also comparison benchmarks.

Random Forest uses the audited features in the notebook, including AgeMissing, CoverageFault, HasPriorClaims and WeekendAccident. Age = 0 becomes missing and is filled using the training median. Preprocessing is learned from the training years only. PolicyNumber, RepNumber, PolicyType, PoliceReportFiled, WitnessPresent, NumberOfSuppliments, AddressChange_Claim and Year are excluded as inputs. Fault remains included but flagged because its availability at claim filing is uncertain.

The lookup rule uses only BasePolicy and Fault. Each claim receives its group's training fraud rate. For an unseen group, it uses the overall training fraud rate.

## 1995 validation comparison

Both approaches trained on 1994 and reviewed 1,039 claims from 1995.

| Metric | Random Forest | Coverage + fault rule |
|---|---:|---:|
| Fraud caught | 151 | 147 |
| Recall at 20% review | 50.17% | 48.84% |
| Precision at 20% review | 14.53% | 14.15% |
| F1 at 20% review | 0.2254 | 0.2194 |
| AP | 0.1179 | 0.1289 |

## 1996 test comparison

Both approaches trained on 1994–1995 and reviewed 817 of the same 4,083 claims. There were 213 actual fraud cases.

| Metric | Random Forest | Coverage + fault rule |
|---|---:|---:|
| Fraud caught | 86 | 88 |
| Recall at 20% review | 40.38% | 41.31% |
| Precision at 20% review | 10.53% | 10.77% |
| F1 at 20% review | 0.1670 | 0.1709 |
| AP | 0.1065 | 0.1031 |
| Legitimate claims reviewed | 731 | 729 |
| Fraud missed | 127 | 125 |

Random review at this capacity would catch about 43 fraud cases on average. Both approaches catch about twice that amount. Accuracy is secondary because always predicting non-fraud gives high accuracy while catching no fraud.

## What we recommend

Keep Random Forest as the original validation-selected model in the project report. The 1996 comparison shows that it does not provide a clear practical improvement over the lookup rule. Lookup caught two more fraud cases, while Random Forest had slightly higher AP.

The lookup rule is a reasonable simpler option because it uses only two inputs and is easier to explain and maintain. This recommendation was made after inspecting 1996 results; it is not a new independent validation result. We should not claim that either approach is reliably better based on this small difference.

## Limitations and next steps

- Many lookup scores are tied. The fixed tie-break affects which claims enter the review queue and the exact fraud count.
- We selected among 18 configurations on 1995, so some validation overfitting is possible.
- We previously inspected 1996. These results must not be described as a completely unseen test, and we should not tune settings or seeds using them.
- Fault's timing still needs confirmation. Both approaches depend on it.
- Most reviewed claims are legitimate, and many fraud cases are missed. Flags should prompt investigation, not automatic rejection.
- Fraud rates change over time. Future performance needs evaluation on newer data.
- SHAP explains model scores, not proof of fraud. Removing Sex alone does not establish fairness.

A useful next step is to collect reliable information available at filing that helps distinguish fraud within the same coverage/fault groups. Any new approach should be evaluated on a later, untouched hold-out set.
