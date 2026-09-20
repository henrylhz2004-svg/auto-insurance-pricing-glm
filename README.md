````markdown
# Auto Insurance Pricing Model Using GLMs

An end-to-end actuarial pricing project that combines **MySQL**, **Python**, and **Generalized Linear Models (GLMs)** to estimate auto insurance claim frequency, claim severity, and policy-level pure premium.

The workflow covers data loading and validation, exploratory analysis, model development, out-of-sample validation, large-loss sensitivity analysis, risk segmentation, and final policy-level pricing output.

---

## Project Overview

The objective of this project is to estimate expected insurance loss cost using the standard frequency–severity framework:

\[
\text{Pure Premium}
=
\text{Expected Claim Frequency}
\times
\text{Expected Claim Severity}
\]

The project models:

- **Claim Frequency** using a Negative Binomial regression model
- **Claim Severity** using a Gamma GLM with a log link
- **Pure Premium** as the product of predicted frequency and predicted severity

The final model produces policy-level expected loss costs and groups policies into Low, Medium, and High pricing risk segments.

---

## Technologies Used

- Python
- MySQL
- pandas
- NumPy
- statsmodels
- scikit-learn
- Matplotlib
- SQLAlchemy
- mysql-connector-python

---

## Data

The project uses two motor insurance datasets:

- `freMTPL2freq.csv` — policy-level exposure and claim count data
- `freMTPL2sev.csv` — claim-level severity data

The main policy variables include:

- Driver age
- Vehicle age
- Vehicle power
- Bonus-malus score
- Vehicle brand
- Fuel type
- Area
- Population density
- Region
- Exposure
- Claim count

The severity dataset contains policy IDs and individual claim amounts.

Raw data files are not included in this repository. Update the file paths in `01_data_import.sql` before importing the data into MySQL.

---

## Database Workflow

The SQL portion of the project handles database creation, data import, validation, and exploratory analysis.

### SQL Files

| File | Purpose |
|---|---|
| `00_create_schema.sql` | Creates the MySQL database and required tables |
| `01_data_import.sql` | Imports the frequency and severity CSV files |
| `02_data_validation.sql` | Performs data-quality and consistency checks |
| `03_data_analysis.sql` | Performs exploratory claim-frequency analysis |

The SQL validation process checks:

- Policy and claim record counts
- Missing or unmatched policy IDs
- Claim-count consistency between the frequency and severity datasets
- Exposure ranges
- Claim amount ranges
- Rating-variable ranges

---

## Python Workflow

| File | Purpose |
|---|---|
| `db_helper.py` | Creates the MySQL connection and loads project data |
| `01_load_data.py` | Reviews dataset structure and basic data quality |
| `02_data_analysis.py` | Performs exploratory frequency analysis and visualization |
| `03_frequency_model.py` | Develops the claim-frequency model |
| `04_frequency_validation.py` | Performs out-of-sample frequency validation |
| `05_severity_model.py` | Develops the claim-severity model |
| `06_severity_validation.py` | Performs out-of-sample severity validation |
| `07_pure_premium.py` | Combines frequency and severity into final pricing results |

---

## Exploratory Analysis

Claim frequency was analyzed across major rating variables including:

- Driver age
- Bonus-malus group
- Vehicle age
- Vehicle power
- Vehicle brand
- Fuel type
- Area
- Population density
- Region

For each segment, claim frequency was calculated as:

\[
\text{Claim Frequency}
=
\frac{\text{Claims}}
{\text{Exposure}}
\]

Frequency relativities were calculated relative to the overall portfolio claim frequency.

### Exploratory Findings

Some major patterns observed during exploratory analysis included:

- Higher bonus-malus groups were associated with substantially higher claim frequency.
- Young drivers showed higher univariate frequency, although much of this effect reduced after controlling for other variables.
- Geographic characteristics were predictive of claim frequency.
- Vehicle characteristics improved model fit when added jointly.

---

# Claim Frequency Model

## Baseline Model

An intercept-only Poisson GLM was first fitted using exposure as an offset.

The exposure adjustment ensures that policies with different observation periods can be compared on an annualized claim-frequency basis.

---

## Model Development

Predictors were added progressively:

1. Driver age and bonus-malus
2. Vehicle characteristics
3. Geographic characteristics

Population density and Area were compared as alternative geographic measures.

For the final frequency specification, **log population density** was preferred over Area because it produced a slightly lower AIC while using fewer parameters.

Region was retained based on a likelihood-ratio test.

---

## Overdispersion

The Poisson model showed substantial overdispersion, with Pearson dispersion materially above 1.

A Negative Binomial model was therefore fitted using the same predictor specification.

The Negative Binomial model produced a lower AIC than the Poisson model and was selected as the final frequency model.

### Final Frequency Predictors

- Driver age group
- Bonus-malus group
- Vehicle age group
- Vehicle power
- Vehicle brand
- Fuel type
- Log population density
- Region

Estimated Negative Binomial dispersion parameter:

\[
\alpha \approx 0.922
\]

---

# Frequency Validation

An 80/20 train-test split was used for out-of-sample validation.

The final Negative Binomial model was fitted on the training set and evaluated on the test set.

Validation included:

- Actual vs predicted total claims
- Actual vs predicted portfolio claim frequency
- Predicted-risk deciles
- Actual vs predicted claim frequency by decile
- Lift by risk decile

The model showed strong risk ordering, with observed claim frequency increasing substantially across predicted-risk deciles.

The highest-risk decile had an observed claim frequency of approximately:

\[
0.30
\]

claims per exposure year, compared with approximately:

\[
0.05
\]

in the lowest-risk decile.

---

# Claim Severity Model

Claim severity was modeled using a **Gamma GLM with a log link**.

Claim records were aggregated to the policy level, and average claim severity was calculated as:

\[
\text{Average Severity}
=
\frac{\text{Total Claim Amount}}
{\text{Number of Claims}}
\]

Policies with multiple claims were weighted using the number of severity records so that policy-level averages based on more claims received greater statistical weight.

---

## Severity Model Development

Predictors were added progressively:

1. Driver age and bonus-malus
2. Vehicle characteristics
3. Geographic characteristics

For severity, **Area** performed better than log population density and was retained in the final model.

Region was also retained based on a likelihood-ratio test.

### Final Severity Predictors

- Driver age group
- Bonus-malus group
- Vehicle age group
- Vehicle power
- Vehicle brand
- Fuel type
- Area
- Region

---

# Large-Loss Sensitivity Analysis

The severity dataset contains a very large claim of approximately:

\[
\$4.08\text{ million}
\]

Because severity models can be sensitive to extreme losses, a sensitivity analysis was performed.

For the 18–24 driver age group, the estimated severity relativity changed from approximately:

- **2.28** using the full dataset
- **1.80** after removing the single largest claim

The effect remained elevated after removing the largest claim, but the decrease showed that the severity model was materially influenced by the extreme loss.

The large claim was retained in the primary model.

The reduced-loss model was used only as a sensitivity diagnostic.

---

# Severity Validation

An 80/20 train-test split was used for severity validation.

The original Gamma model produced:

- Actual test-set average severity: approximately **$2,097.57**
- Predicted test-set average severity: approximately **$2,219.67**
- Calibration difference: approximately **+5.82%**

The largest claim was located in the training set.

After refitting the severity model without the single largest training loss:

- Predicted average severity: approximately **$2,095.07**
- Calibration difference: approximately **-0.12%**

This result demonstrates the sensitivity of severity estimates to extreme losses.

Risk-decile analysis showed noisier ordering than the frequency model, but the highest predicted-severity decile still contained substantially higher observed claim severity.

---

# Pure Premium Model

The final pricing model combines the two components:

\[
\text{Pure Premium}
=
\text{Predicted Frequency}
\times
\text{Predicted Severity}
\]

For each policy:

- `predicted_frequency` = expected claims per exposure year
- `predicted_severity` = expected loss per claim
- `pure_premium` = expected annual loss cost
- `predicted_loss` = pure premium multiplied by observed exposure

---

## Portfolio Results

The final full-portfolio model produced approximately:

- **Predicted claim frequency:** 0.1022
- **Predicted total claims:** 36,640
- **Predicted total loss:** $80.6 million
- **Portfolio pure premium:** $224.90 per exposure year

---

# Pricing Risk Segments

Policies were ranked by predicted pure premium and divided into ten risk deciles.

The deciles were then grouped into three broader pricing segments:

- **Low Risk:** deciles 1–4
- **Medium Risk:** deciles 5–7
- **High Risk:** deciles 8–10

Approximate predicted pure premiums were:

| Risk Segment | Predicted Pure Premium |
|---|---:|
| Low | $114 |
| Medium | $185 |
| High | $483 |

Both predicted claim frequency and predicted claim severity increased across the broader risk segments.

---

# Pure Premium Diagnostic

A matched subset was created using policies for which the claim count in the policy dataset matched the number of available severity records.

This retained approximately:

- **98.66% of policies**
- **98.72% of exposure**

Within this subset, predicted loss cost was higher than observed loss cost.

However, this comparison should **not** be interpreted as an unbiased portfolio calibration measure.

Although the excluded policies represent only a small share of total policy records, severity records are disproportionately missing among claiming policies.

Approximately **9,657 reported claims** do not have corresponding severity records in the available severity dataset.

For this reason, the matched-subset actual-vs-predicted pure premium comparison is treated as a diagnostic rather than formal validation.

Formal out-of-sample validation is performed separately for the frequency and severity models in:

- `04_frequency_validation.py`
- `06_severity_validation.py`

---

# Output Files

Running the project generates analytical outputs including:

```text
driver_age_frequency.png
bonus_malus_frequency.png
vehicle_age_frequency.png
area_frequency.png
vehicle_brand_frequency.png

frequency_decile_validation.png
severity_decile_validation.png

pure_premium_decile_diagnostic.png
pure_premium_by_risk_segment.png

policy_pricing_results.csv
````

The final CSV contains policy-level pricing results including:

* Predicted claim frequency
* Predicted claim severity
* Pure premium
* Expected loss
* Pure premium risk decile
* Low / Medium / High risk segment

---

# Project Structure

```text
auto-insurance-pricing/
│
├── 00_create_schema.sql
├── 01_data_import.sql
├── 02_data_validation.sql
├── 03_data_analysis.sql
│
├── db_helper.py
├── 01_load_data.py
├── 02_data_analysis.py
├── 03_frequency_model.py
├── 04_frequency_validation.py
├── 05_severity_model.py
├── 06_severity_validation.py
├── 07_pure_premium.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

# Setup

## 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

---

## 2. Configure MySQL

Run the SQL files in order:

```text
00_create_schema.sql
01_data_import.sql
02_data_validation.sql
03_data_analysis.sql
```

Before running `01_data_import.sql`, replace:

```text
/path/to/freMTPL2freq.csv
/path/to/freMTPL2sev.csv
```

with the local paths to the data files.

---

## 3. Configure Database Credentials

The Python scripts read the MySQL password from the `MYSQL_PASSWORD` environment variable.

### Windows PowerShell

```powershell
$env:MYSQL_PASSWORD="your_password"
```

Optional environment variables include:

```text
MYSQL_USER
MYSQL_HOST
MYSQL_PORT
MYSQL_DATABASE
```

If they are not provided, the project defaults to:

```text
User: root
Host: localhost
Port: 3306
Database: insurance_project
```

---

## 4. Run the Python Workflow

Run the scripts in order:

```text
01_load_data.py
02_data_analysis.py
03_frequency_model.py
04_frequency_validation.py
05_severity_model.py
06_severity_validation.py
07_pure_premium.py
```

---

# Key Actuarial Concepts Demonstrated

This project demonstrates several practical insurance-pricing concepts:

* Exposure-adjusted claim-frequency modeling
* Detection and treatment of overdispersion
* Negative Binomial frequency modeling
* Gamma severity modeling with a log link
* Frequency and severity relativities
* Likelihood-ratio testing for groups of rating factors
* Out-of-sample model validation
* Risk-decile analysis
* Lift analysis
* Large-loss sensitivity analysis
* Frequency × severity pure premium construction
* Policy-level expected loss estimation
* Risk segmentation for pricing applications

---

# Limitations

Several limitations should be considered when interpreting the results:

* The available severity dataset does not contain a claim amount record for every claim reported in the policy-frequency dataset.
* Large claims have a material effect on severity estimates.
* The project models pure premium only and does not include expenses, profit margins, taxes, reinsurance costs, or other commercial loadings required to produce a final charged premium.
* The Low / Medium / High segments are model-based expected-loss groups rather than underwriting decisions or customer classifications.
* Additional validation across time periods or independent datasets would be required before production use.

---

# Disclaimer

This project is intended for educational and portfolio purposes.

The results should not be interpreted as production insurance rates.

