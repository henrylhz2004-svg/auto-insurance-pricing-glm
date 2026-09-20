import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from db_helper import load_policy_data, load_claim_data

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_rows", None)

# VALIDATION SETTINGS
TEST_SIZE = 0.20
RANDOM_STATE = 42
N_DECILES = 10

# LOAD SOURCE DATA
policy_df = load_policy_data()
claims_df = load_claim_data()

# VALIDATE MODELING INPUTS
assert (claims_df["ClaimAmount"] > 0).all(), (
    "Gamma severity modeling requires strictly positive claim amounts."
)

# SELECT POLICY-LEVEL RISK FEATURES
policy_features = policy_df[
    [
        "IDpol",
        "ClaimNb",
        "Area",
        "VehPower",
        "VehAge",
        "DrivAge",
        "BonusMalus",
        "VehBrand",
        "VehGas",
        "Region"
    ]
].copy()

# CREATE CATEGORICAL RISK GROUPS
def add_risk_groups(data):
    data = data.copy()

    data["driver_age_group"] = pd.cut(
        data["DrivAge"],
        bins=[17, 24, 34, 49, 64, float("inf")],
        labels=["18-24", "25-34", "35-49", "50-64", "65+"]
    )

    data["bonus_malus_group"] = pd.cut(
        data["BonusMalus"],
        bins=[0, 50, 75, 100, 125, float("inf")],
        labels=["50", "51-75", "76-100", "101-125", "126+"]
    )

    data["vehicle_age_group"] = pd.cut(
        data["VehAge"],
        bins=[-1, 0, 5, 10, 15, 20, float("inf")],
        labels=["0", "1-5", "6-10", "11-15", "16-20", "21+"]
    )

    return data

# AGGREGATE CLAIMS AND BUILD POLICY-LEVEL SEVERITY DATA
def build_severity_data(
    claim_data,
    valid_policy_ids=None,
    validate_claim_counts=True
):
    if valid_policy_ids is not None:
        claim_data = claim_data[
            claim_data["IDpol"].isin(valid_policy_ids)
        ].copy()

    claim_totals = (
        claim_data.groupby("IDpol", as_index=False)
        .agg(
            total_claim_amount=("ClaimAmount", "sum"),
            severity_claim_count=("ClaimAmount", "count")
        )
    )

    severity_data = claim_totals.merge(
        policy_features,
        on="IDpol",
        how="inner"
    )

    mismatch_count = 0

    if validate_claim_counts:
        mismatch = (
            severity_data["severity_claim_count"]
            != severity_data["ClaimNb"]
        )

        mismatch_count = int(mismatch.sum())
        severity_data = severity_data.loc[~mismatch].copy()

    severity_data["average_severity"] = (
        severity_data["total_claim_amount"]
        / severity_data["severity_claim_count"]
    )

    severity_data = add_risk_groups(severity_data)

    return severity_data, mismatch_count

# FIT A CLAIM-COUNT-WEIGHTED GAMMA GLM
def fit_gamma(formula, data):
    return smf.glm(
        formula=formula,
        data=data,
        family=sm.families.Gamma(
            link=sm.families.links.Log()
        ),
        var_weights=data["severity_claim_count"]
    ).fit()

# CALCULATE PORTFOLIO-LEVEL SEVERITY CALIBRATION
def calculate_calibration(data, predicted_total_column):
    actual_total = data["total_claim_amount"].sum()
    predicted_total = data[predicted_total_column].sum()
    claim_count = data["severity_claim_count"].sum()

    actual_average = actual_total / claim_count
    predicted_average = predicted_total / claim_count

    error_pct = (
        (predicted_total - actual_total)
        / actual_total
        * 100
    )

    return {
        "actual_total": actual_total,
        "predicted_total": predicted_total,
        "claim_count": claim_count,
        "actual_average": actual_average,
        "predicted_average": predicted_average,
        "error_pct": error_pct
    }

# BUILD THE MAIN SEVERITY DATASET
severity_df, mismatch_count = build_severity_data(
    claims_df,
    validate_claim_counts=True
)

print("\nSEVERITY DATA QUALITY")
print("Policies with claim-count mismatch:", mismatch_count)
print("Severity policies used:", len(severity_df))


# IDENTIFY THE LARGEST INDIVIDUAL CLAIM
largest_claim_idx = claims_df["ClaimAmount"].idxmax()
largest_claim = claims_df.loc[largest_claim_idx]
largest_policy_id = int(largest_claim["IDpol"])
largest_claim_amount = float(largest_claim["ClaimAmount"])

# SPLIT DATA INTO TRAINING AND TEST SETS
train_df, test_df = train_test_split(
    severity_df,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

test_df = test_df.copy()

largest_loss_in_test = (
    largest_policy_id in test_df["IDpol"].values
)

print("\nTRAIN / TEST SPLIT")
print("Training severity policies:", len(train_df))
print("Testing severity policies:", len(test_df))
print("Training claims:", int(train_df["severity_claim_count"].sum()))
print("Testing claims:", int(test_df["severity_claim_count"].sum()))
print("Largest-loss policy in test set:", largest_loss_in_test)

# DEFINE THE FINAL SEVERITY MODEL SPECIFICATION
severity_formula = """
    average_severity ~
    C(driver_age_group, Treatment(reference="35-49"))
    + C(bonus_malus_group, Treatment(reference="50"))
    + C(vehicle_age_group, Treatment(reference="6-10"))
    + C(VehPower)
    + C(VehBrand)
    + C(VehGas)
    + C(Area)
    + C(Region)
"""

# FIT THE FINAL GAMMA MODEL ON TRAINING DATA
severity_train_model = fit_gamma(
    severity_formula,
    train_df
)

# GENERATE OUT-OF-SAMPLE TEST-SET PREDICTIONS
test_df["predicted_severity"] = (
    severity_train_model.predict(test_df)
)

test_df["predicted_total_claim_amount"] = (
    test_df["predicted_severity"]
    * test_df["severity_claim_count"]
)

# CALCULATE OVERALL TEST-SET CALIBRATION
main_calibration = calculate_calibration(
    test_df,
    "predicted_total_claim_amount"
)

actual_total_amount = main_calibration["actual_total"]
predicted_total_amount = main_calibration["predicted_total"]
total_claims = main_calibration["claim_count"]
actual_severity = main_calibration["actual_average"]
predicted_severity = main_calibration["predicted_average"]
calibration_error_pct = main_calibration["error_pct"]

print("\nTEST SET SEVERITY CALIBRATION")
print(f"Actual total claim amount: {actual_total_amount:,.2f}")
print(f"Predicted total claim amount: {predicted_total_amount:,.2f}")
print(f"Actual average severity: {actual_severity:,.2f}")
print(f"Predicted average severity: {predicted_severity:,.2f}")
print(f"Calibration error (%): {calibration_error_pct:.2f}")

# ASSIGN TEST POLICIES TO PREDICTED-SEVERITY DECILES
test_df["severity_decile"] = pd.qcut(
    test_df["predicted_severity"].rank(method="first"),
    q=N_DECILES,
    labels=False
) + 1

# SUMMARIZE ACTUAL AND PREDICTED SEVERITY BY RISK DECILE
decile_summary = (
    test_df.groupby(
        "severity_decile",
        observed=True
    )
    .agg(
        policies=("IDpol", "count"),
        claims=("severity_claim_count", "sum"),
        actual_total_amount=("total_claim_amount", "sum"),
        predicted_total_amount=("predicted_total_claim_amount", "sum")
    )
)

decile_summary["actual_severity"] = (
    decile_summary["actual_total_amount"]
    / decile_summary["claims"]
)

decile_summary["predicted_severity"] = (
    decile_summary["predicted_total_amount"]
    / decile_summary["claims"]
)

decile_summary["actual_lift"] = (
    decile_summary["actual_severity"]
    / actual_severity
)

decile_summary["predicted_lift"] = (
    decile_summary["predicted_severity"]
    / predicted_severity
)

print("\nSEVERITY DECILE VALIDATION")
print(decile_summary)

# PLOT ACTUAL VS PREDICTED SEVERITY BY RISK DECILE
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(
    decile_summary.index,
    decile_summary["actual_severity"],
    marker="o",
    label="Actual"
)

ax.plot(
    decile_summary.index,
    decile_summary["predicted_severity"],
    marker="o",
    label="Predicted"
)

ax.set_xlabel("Severity Risk Decile")
ax.set_ylabel("Average Claim Severity")
ax.set_title("Actual vs Predicted Claim Severity by Risk Decile")
ax.set_xticks(decile_summary.index)
ax.legend()

fig.tight_layout()

fig.savefig(
    "severity_decile_validation.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# LARGE-LOSS SENSITIVITY VALIDATION
print("\nLARGE-LOSS SENSITIVITY CHECK")
print(f"Largest claim amount: {largest_claim_amount:,.2f}")
print("Largest-loss policy ID:", largest_policy_id)
print(
    "Largest-loss policy in training set:",
    largest_policy_id in train_df["IDpol"].values
)

if largest_policy_id in train_df["IDpol"].values:
    train_policy_ids = train_df["IDpol"]

    sensitivity_claims = claims_df[
        claims_df["IDpol"].isin(train_policy_ids)
    ].copy()

    # Remove only the single largest claim, not the entire policy.
    sensitivity_claims = sensitivity_claims.drop(
        index=largest_claim_idx
    )

    # Claim-count validation is intentionally disabled because one observed
    # claim has been removed for this sensitivity test.
    train_df_sensitivity, _ = build_severity_data(
        sensitivity_claims,
        valid_policy_ids=train_policy_ids,
        validate_claim_counts=False
    )

    severity_train_sensitivity = fit_gamma(
        severity_formula,
        train_df_sensitivity
    )

    test_df["predicted_severity_sensitivity"] = (
        severity_train_sensitivity.predict(test_df)
    )

    test_df["predicted_total_amount_sensitivity"] = (
        test_df["predicted_severity_sensitivity"]
        * test_df["severity_claim_count"]
    )

    sensitivity_calibration = calculate_calibration(
        test_df,
        "predicted_total_amount_sensitivity"
    )

    print("\nLARGE-LOSS SENSITIVITY VALIDATION")
    print(f"Actual average severity: {actual_severity:,.2f}")
    print(f"Original predicted severity: {predicted_severity:,.2f}")
    print(
        "Without largest training loss: "
        f"{sensitivity_calibration['predicted_average']:,.2f}"
    )
    print(
        "Original calibration error (%): "
        f"{calibration_error_pct:.2f}"
    )
    print(
        "Sensitivity calibration error (%): "
        f"{sensitivity_calibration['error_pct']:.2f}"
    )
else:
    print(
        "Sensitivity refit skipped because the largest claim "
        "is not in the training set."
    )






