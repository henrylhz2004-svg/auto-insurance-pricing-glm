import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from db_helper import load_policy_data

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_rows", None)

# VALIDATION SETTINGS
TEST_SIZE = 0.20
RANDOM_STATE = 42
N_DECILES = 10

# LOAD POLICY DATA
df = load_policy_data()

# VALIDATE EXPOSURE
assert (df["Exposure"] > 0).all(), "Exposure contains zero or negative values."

# CREATE CATEGORICAL RISK GROUPS
df["driver_age_group"] = pd.cut(
    df["DrivAge"],
    bins=[17, 24, 34, 49, 64, float("inf")],
    labels=["18-24", "25-34", "35-49", "50-64", "65+"]
)

df["bonus_malus_group"] = pd.cut(
    df["BonusMalus"],
    bins=[0, 50, 75, 100, 125, float("inf")],
    labels=["50", "51-75", "76-100", "101-125", "126+"]
)

df["vehicle_age_group"] = pd.cut(
    df["VehAge"],
    bins=[-1, 0, 5, 10, 15, 20, float("inf")],
    labels=["0", "1-5", "6-10", "11-15", "16-20", "21+"]
)

df["log_density"] = np.log(df["Density"])

# DEFINE FINAL FREQUENCY MODEL SPECIFICATION
frequency_formula = """
    ClaimNb ~
    C(driver_age_group, Treatment(reference="35-49"))
    + C(bonus_malus_group, Treatment(reference="50"))
    + C(vehicle_age_group, Treatment(reference="6-10"))
    + C(VehPower)
    + C(VehBrand)
    + C(VehGas)
    + log_density
    + C(Region)
"""

# SPLIT DATA INTO TRAINING AND TEST SETS
train_df, test_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

print("Training policies:", len(train_df))
print("Testing policies:", len(test_df))

# FIT POISSON MODEL TO OBTAIN STABLE STARTING VALUES
poisson_train = smf.glm(
    formula=frequency_formula,
    data=train_df,
    family=sm.families.Poisson(),
    offset=np.log(train_df["Exposure"])
).fit()

# FIT NEGATIVE BINOMIAL MODEL ON TRAINING DATA
nb_train_model = smf.negativebinomial(
    formula=frequency_formula,
    data=train_df,
    exposure=train_df["Exposure"]
)

start_params = np.append(
    poisson_train.params.to_numpy(),
    1.0
)

nb_train = nb_train_model.fit(
    start_params=start_params,
    method="bfgs",
    maxiter=200,
    disp=False
)

print("\nMODEL FIT CHECK")
print("Converged:", nb_train.mle_retvals["converged"])
print("Alpha:", nb_train.params["alpha"])

# GENERATE OUT-OF-SAMPLE PREDICTIONS
test_df = test_df.copy()

test_df["predicted_claims"] = nb_train.predict(
    test_df,
    exposure=test_df["Exposure"]
)

test_df["predicted_frequency"] = (
    test_df["predicted_claims"]
    / test_df["Exposure"]
)

# CALCULATE PORTFOLIO-LEVEL TEST-SET CALIBRATION
actual_claims = test_df["ClaimNb"].sum()
predicted_claims = test_df["predicted_claims"].sum()
total_test_exposure = test_df["Exposure"].sum()

actual_frequency = actual_claims / total_test_exposure
predicted_frequency = predicted_claims / total_test_exposure

calibration_error_pct = (
    (predicted_claims - actual_claims)
    / actual_claims
    * 100
)

print("\nTEST SET CALIBRATION")
print("Actual claims:", actual_claims)
print("Predicted claims:", predicted_claims)
print("Actual frequency:", actual_frequency)
print("Predicted frequency:", predicted_frequency)
print("Claim count calibration error (%):", calibration_error_pct)

# ASSIGN TEST POLICIES TO PREDICTED-RISK DECILES
test_df["risk_decile"] = pd.qcut(
    test_df["predicted_frequency"].rank(method="first"),
    q=N_DECILES,
    labels=False
) + 1

# SUMMARIZE ACTUAL AND PREDICTED PERFORMANCE BY RISK DECILE
decile_summary = (
    test_df.groupby("risk_decile")
    .agg(
        policies=("IDpol", "count"),
        exposure=("Exposure", "sum"),
        actual_claims=("ClaimNb", "sum"),
        predicted_claims=("predicted_claims", "sum")
    )
)

decile_summary["actual_frequency"] = (
    decile_summary["actual_claims"]
    / decile_summary["exposure"]
)

decile_summary["predicted_frequency"] = (
    decile_summary["predicted_claims"]
    / decile_summary["exposure"]
)

# CALCULATE ACTUAL AND PREDICTED LIFT RELATIVE TO PORTFOLIO AVERAGES
decile_summary["actual_lift"] = (
    decile_summary["actual_frequency"]
    / actual_frequency
)

decile_summary["predicted_lift"] = (
    decile_summary["predicted_frequency"]
    / predicted_frequency
)

print("\nRISK DECILE VALIDATION")
print(decile_summary)

# PLOT ACTUAL VS PREDICTED FREQUENCY BY RISK DECILE
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(
    decile_summary.index,
    decile_summary["actual_frequency"],
    marker="o",
    label="Actual"
)

ax.plot(
    decile_summary.index,
    decile_summary["predicted_frequency"],
    marker="o",
    label="Predicted"
)

ax.set_xlabel("Risk Decile")
ax.set_ylabel("Claims per Exposure Year")
ax.set_title("Actual vs Predicted Claim Frequency by Risk Decile")
ax.set_xticks(decile_summary.index)
ax.legend()

fig.tight_layout()

fig.savefig(
    "frequency_decile_validation.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()









