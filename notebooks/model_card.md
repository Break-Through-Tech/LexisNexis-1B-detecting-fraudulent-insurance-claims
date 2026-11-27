# Model card: Vehicle insurance claim fraud detection

**Project:** Break Through Tech AI Studio — LexisNexis Team 1B  
**Prepared by:** Mohammad Ikbal, Emanuel Messele, Linh Le, Mukundan Thanigaivalen, Princy Maheshwari  
**Updated:** November 27, 2026  
**Status:** Educational prototype; not ready for production use

## Model overview

The model ranks vehicle insurance claims so investigators can prioritize claims for review. The target is `FraudFound_P`: 1 means a recorded fraud case and 0 means a recorded non-fraud case.

Random Forest was the original model selected using 1995 validation results. We also evaluated a simpler coverage + fault lookup rule. The lookup rule gives each claim the fraud rate of its BasePolicy and Fault group, calculated from training data. An unseen group receives the overall training fraud rate.

Both approaches performed similarly on 1996. Random Forest remains the original validation-selected model in our report; the lookup rule is a reasonable simpler option to consider. That recommendation was made after seeing the test comparison.

## Intended use

The intended use is to help prioritize human investigation when investigators can review only 20% of claims. Scores are ranked from highest to lowest, and the top 20% are flagged, rounding up to a whole claim. Equal scores are ordered using the notebook's fixed pandas shuffle with seed 42.

A flag means “review this claim,” not “this claim is fraudulent.” The model should not automatically reject claims, determine guilt, set insurance prices, or replace an investigator's judgment. Its scores have not been checked for calibration, so they should not be treated as verified probabilities of fraud.

## Data and evaluation setup

The project uses the Kaggle Vehicle Insurance Claim Fraud Detection dataset, containing claims from 1994–1996. Fraud is uncommon, making up about 6% of the data. The data dictionary provides working definitions, but the timing of some fields still needs confirmation.

The notebook removes one row with invalid claim month and weekday values, leaving 15,419 claims.

| Stage | Years | Claims | Fraud cases |
|---|---|---:|---:|
| Initial training | 1994 | 6,141 | 409 |
| Validation and model selection | 1995 | 5,195 | 301 |
| Combined training for final evaluation | 1994–1995 | 11,336 | 710 |
| Test comparison | 1996 | 4,083 | 213 |

We used a temporal hold-out, not K-Fold cross-validation. Model settings were compared using 1995. The chosen Random Forest and lookup rule were then fitted using 1994–1995 and evaluated on the same 1996 claims.

**Prior test exposure:** 1996 had already been inspected in earlier project work. This comparison is not a completely unseen final test. We did not change model settings or the tie-break seed to improve these reported results.

## Inputs and preprocessing

Random Forest uses the retained policy, claimant, vehicle and accident features documented in the notebook. These include coverage, Fault, age, claim history, vehicle information and timing categories.

| Step | What we did |
|---|---|
| Missing age | Treat Age = 0 as missing, add AgeMissing, and fill age using the training median. |
| Rare vehicle makes | Group makes with fewer than 50 training claims as Other; reuse that training-derived list for evaluation. |
| Ordered categories | Convert the specified ordered buckets into numeric ranks, not estimated ages or amounts. |
| Other categories | One-hot encode them and align evaluation columns to the training columns. |
| Added features | CoverageFault combines coverage and fault; HasPriorClaims indicates previous claims; WeekendAccident indicates Saturday or Sunday. |
| Training-only fitting | Learn preprocessing from the training years. Refit it on 1994–1995 for the final comparison. |

PolicyNumber, RepNumber, PolicyType, PoliceReportFiled, WitnessPresent, NumberOfSuppliments, AddressChange_Claim and Year are excluded as model inputs. Year is used for splitting the data.

Fault is included but remains a potential leakage risk because its availability when a claim is first filed is uncertain. AgeOfPolicyHolder is retained as a category; its youngest bucket is not used to fill missing age. Sex is retained in the main model, with a separate removal diagnostic.

## Training and selection

We compared logistic regression, Random Forest and gradient boosting. Each family used two settings and three imbalance approaches: no adjustment, balanced weights and SMOTENC. This produced 18 candidates. Synthetic sampling was applied only to training data, never validation or test data.

The selected Random Forest uses:

- 200 trees.
- Maximum tree depth of 6.
- At least 10 samples per leaf.
- No class weights and no SMOTE.
- Random seed 42.

The selection criterion was the most fraud captured while reviewing 20% of 1995 claims, with Average Precision (AP) as the tie-breaker. Random Forest caught 151 fraud cases versus 147 for the lookup rule, supporting the original selection. The difference was small, and lookup had higher validation AP.

## Performance

Recall measures the share of actual fraud caught. Precision measures the share of reviewed claims that are fraud. F1 balances recall and precision. AP summarizes ranking quality across score thresholds; it is not accuracy.

### 1995 validation

Both approaches reviewed 1,039 claims after training on 1994.

| Metric | Random Forest | Lookup rule |
|---|---:|---:|
| Fraud caught | 151 of 301 | 147 of 301 |
| Recall | 50.17% | 48.84% |
| Precision | 14.53% | 14.15% |
| F1 | 0.2254 | 0.2194 |
| AP | 0.1179 | 0.1289 |

### 1996 test comparison

Both approaches reviewed 817 claims after training on 1994–1995.

| Metric | Random Forest | Lookup rule |
|---|---:|---:|
| Fraud caught | 86 of 213 | 88 of 213 |
| Recall | 40.38% | 41.31% |
| Precision | 10.53% | 10.77% |
| F1 | 0.1670 | 0.1709 |
| AP | 0.1065 | 0.1031 |
| Accuracy at the review decision | 78.99% | 79.08% |

Random review of 817 claims would catch about 43 fraud cases on average, with about 20% recall and 5.22% precision. Both approaches captured about twice that expected fraud count.

Always predicting non-fraud would achieve about 94.78% accuracy but catch no fraud. This is why accuracy is not our main measure.

### Types of mistakes on 1996

| Outcome | Random Forest | Lookup rule |
|---|---:|---:|
| Fraud reviewed: true positives | 86 | 88 |
| Legitimate claims reviewed: false positives | 731 | 729 |
| Fraud not reviewed: false negatives | 127 | 125 |
| Legitimate claims not reviewed: true negatives | 3,139 | 3,141 |

False positives create investigation work and may inconvenience legitimate claimants. False negatives leave fraudulent claims outside the review queue. Neither error type was assigned a monetary cost in this project.

The two-case advantage for lookup does not establish reliable superiority. Lookup produces many tied scores, so the exact fraud count depends partly on the fixed tie-break. Random Forest had slightly higher test AP. We did not calculate confidence intervals or test statistical significance.

## Explanations and feature checks

The notebook uses SHAP to explain Random Forest scores. Contributions from one-hot columns are grouped back to their source features. It includes examples of caught fraud, a legitimate claim flagged by mistake, and missed fraud.

Similar coverage and fault information can raise scores for both fraudulent and legitimate claims. SHAP explains how the model produced a score; it does not show that a feature causes fraud or that a claim is fraudulent.

The following Random Forest diagnostics use the same 1996 review capacity:

| Diagnostic | Fraud caught | Recall |
|---|---:|---:|
| Original model | 86 | 40.38% |
| Remove Fault and CoverageFault | 54 | 25.35% |
| Remove Sex | 87 | 40.85% |

Removing Fault and its engineered combination substantially reduces capture, showing the model's dependence on this information. This does not establish that Fault is safe to use. Removing Sex gives similar overall performance, but does not prove fairness across groups.

## Limitations

- **Uncertain feature timing:** Fault may be determined after filing. Confirm its availability before any intake-time use.
- **Historical data:** claims from 1994–1996 may not represent current insurance practices or fraud patterns.
- **Changing fraud rates:** prevalence declines from about 6.7% in 1994 to 5.2% in 1996, and validation performance does not carry over exactly to testing.
- **Limited added value from complexity:** Random Forest did not show a clear practical improvement over the two-input lookup rule at this review capacity.
- **Selection bias:** comparing 18 configurations on one validation year can overfit that year. Prior test exposure further limits the strength of the final evaluation.
- **Label quality:** the project has not independently verified the recorded fraud labels. Undetected fraud may be recorded as non-fraud.
- **Fairness:** subgroup error rates and impacts have not been fully evaluated. Other features can still act as demographic proxies if Sex is removed.
- **Operational assumptions:** 20% review capacity is a project assumption. Investigation costs, staffing needs and financial benefits were not measured.

## Before practical use

Confirm feature timing, review label quality and test on newer claims that have not influenced model development. Evaluate subgroup recall and false-positive rates, check whether scores reflect actual fraud frequencies, and agree on a review capacity with investigators.

If a pilot is approved, keep human review, monitor fraud capture and false positives, and watch for changes in input data and fraud rates. Reassess performance when those patterns change. A useful research direction is collecting reliable filing-time information that helps distinguish claims within the same coverage/fault groups.

## Project conclusion

Random Forest was a reasonable model to test and was selected using the original validation rule. The later comparison showed that a simple coverage + fault lookup rule achieved similar results with less complexity. Both can prioritize claims better than random review in this dataset, but neither is ready for automatic claim decisions.
