from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

from db_helper import load_policy_data, load_claim_data

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_rows", None)

# OUTPUT SETTINGS
OUTPUT_DIR = Path(__file__).resolve().parent
N_DECILES = 10

# LOAD SOURCE DATA
policy_df = load_policy_data()
claims_df = load_claim_data()

# VALIDATE MODELING INPUTS
assert (policy_df["Exposure"] > 0).all(), (
    "Exposure contains zero or negative values."
)
assert (policy_df["Density"] > 0).all(), (
    "Density contains zero or negative values."
)
assert (claims_df["ClaimAmount"] > 0).all(), (
    "Claim amounts must be strictly positive."
)

# CHECK FREQUENCY-SEVERITY DATA CONSISTENCY
actual_claim_count = int(policy_df["ClaimNb"].sum())
severity_claim_count = len(claims_df)

positive_claim_policy_ids = set(
    policy_df.loc[policy_df["ClaimNb"] > 0, "IDpol"]
)
severity_policy_ids = set(claims_df["IDpol"])

missing_severity_policy_ids = (
    positive_claim_policy_ids - severity_policy_ids
)
extra_severity_policy_ids = (
    severity_policy_ids - positive_claim_policy_ids
)

missing_severity_claims = int(
    policy_df.loc[
        policy_df["IDpol"].isin(missing_severity_policy_ids),
        "ClaimNb"
    ].sum()
)

print("\nFREQUENCY-SEVERITY DATA CONSISTENCY")
print("Claims recorded in policy data:", actual_claim_count)
print("Claim records in severity data:", severity_claim_count)
print(
    "Claiming policies missing severity records:",
    len(missing_severity_policy_ids)
)
print(
    "Claims associated with missing severity policies:",
    missing_severity_claims
)
print(
    "Severity policies not marked as claiming:",
    len(extra_severity_policy_ids)
)

# CREATE MODELING VARIABLES
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

    data["log_density"] = np.log(data["Density"])

    return data

policy_df = add_risk_groups(policy_df)

# BUILD POLICY-LEVEL SEVERITY DATA
def build_severity_data(policy_data, claim_data):
    policy_features = policy_data[
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
            "Density",
            "Region",
            "driver_age_group",
            "bonus_malus_group",
            "vehicle_age_group"
        ]
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

    return severity_data, mismatch_count

severity_df, severity_mismatch_count = build_severity_data(
    policy_df,
    claims_df
)

print("\nSEVERITY MODELING DATA")
print("Policies with claim-count mismatch:", severity_mismatch_count)
print("Severity policies used:", len(severity_df))

# DEFINE FINAL FREQUENCY MODEL
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

# DEFINE FINAL SEVERITY MODEL
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

# FIT FINAL NEGATIVE BINOMIAL FREQUENCY MODEL
def fit_frequency_model(data):
    poisson_model = smf.glm(
        formula=frequency_formula,
        data=data,
        family=sm.families.Poisson(),
        offset=np.log(data["Exposure"])
    ).fit()

    nb_model = smf.negativebinomial(
        formula=frequency_formula,
        data=data,
        exposure=data["Exposure"]
    )

    start_params = np.append(
        poisson_model.params.to_numpy(),
        1.0
    )

    return nb_model.fit(
        start_params=start_params,
        method="bfgs",
        maxiter=200,
        disp=False
    )

frequency_model = fit_frequency_model(policy_df)

# FIT FINAL GAMMA SEVERITY MODEL
def fit_severity_model(data):
    return smf.glm(
        formula=severity_formula,
        data=data,
        family=sm.families.Gamma(
            link=sm.families.links.Log()
        ),
        var_weights=data["severity_claim_count"]
    ).fit()

severity_model = fit_severity_model(severity_df)

# CREATE FULL-PORTFOLIO PRICING DATASET
pricing_df = policy_df.copy()

# PREDICT FREQUENCY, SEVERITY, PURE PREMIUM, AND EXPECTED LOSS
pricing_df["predicted_claims"] = frequency_model.predict(
    pricing_df,
    exposure=pricing_df["Exposure"]
)

pricing_df["predicted_frequency"] = (
    pricing_df["predicted_claims"]
    / pricing_df["Exposure"]
)

pricing_df["predicted_severity"] = (
    severity_model.predict(pricing_df)
)

pricing_df["pure_premium"] = (
    pricing_df["predicted_frequency"]
    * pricing_df["predicted_severity"]
)

pricing_df["predicted_loss"] = (
    pricing_df["pure_premium"]
    * pricing_df["Exposure"]
)

# PORTFOLIO-LEVEL PRICING SUMMARY
total_exposure = pricing_df["Exposure"].sum()
portfolio_predicted_claims = pricing_df["predicted_claims"].sum()
portfolio_predicted_loss = pricing_df["predicted_loss"].sum()

portfolio_frequency = (
    portfolio_predicted_claims / total_exposure
)
portfolio_pure_premium = (
    portfolio_predicted_loss / total_exposure
)

print("\nFINAL MODEL CHECK")
print(
    "Frequency model converged:",
    frequency_model.mle_retvals["converged"]
)
print(
    "Frequency alpha:",
    frequency_model.params["alpha"]
)

print("\nPORTFOLIO PURE PREMIUM SUMMARY")
print(f"Predicted claim frequency: {portfolio_frequency:.6f}")
print(f"Predicted total claims: {portfolio_predicted_claims:,.2f}")
print(f"Predicted total loss: {portfolio_predicted_loss:,.2f}")
print(f"Portfolio pure premium: {portfolio_pure_premium:,.2f}")

# BUILD MATCHED SUBSET FOR ACTUAL-LOSS DIAGNOSTICS
claim_amount_summary = (
    claims_df.groupby("IDpol", as_index=False)
    .agg(
        actual_total_loss=("ClaimAmount", "sum"),
        severity_claim_count=("ClaimAmount", "count")
    )
)

pure_premium_diagnostic_df = pricing_df.merge(
    claim_amount_summary,
    on="IDpol",
    how="left"
)

pure_premium_diagnostic_df["severity_claim_count"] = (
    pure_premium_diagnostic_df["severity_claim_count"].fillna(0)
)
pure_premium_diagnostic_df["actual_total_loss"] = (
    pure_premium_diagnostic_df["actual_total_loss"].fillna(0)
)

pure_premium_diagnostic_df["complete_loss_record"] = (
    pure_premium_diagnostic_df["severity_claim_count"]
    == pure_premium_diagnostic_df["ClaimNb"]
)

pure_premium_diagnostic_df = (
    pure_premium_diagnostic_df.loc[
        pure_premium_diagnostic_df["complete_loss_record"]
    ].copy()
)

# CHECK MATCHED-SUBSET COVERAGE
validation_policy_count = len(pure_premium_diagnostic_df)
validation_exposure = pure_premium_diagnostic_df["Exposure"].sum()

policy_coverage_pct = (
    validation_policy_count / len(pricing_df) * 100
)
exposure_coverage_pct = (
    validation_exposure / total_exposure * 100
)

print("\nMATCHED-SUBSET COVERAGE")
print("Policies retained:", validation_policy_count)
print(f"Policy coverage (%): {policy_coverage_pct:.2f}")
print(f"Exposure coverage (%): {exposure_coverage_pct:.2f}")

# CALCULATE MATCHED-SUBSET PURE PREMIUM DIAGNOSTIC
validation_actual_loss = (
    pure_premium_diagnostic_df["actual_total_loss"].sum()
)
validation_predicted_loss = (
    pure_premium_diagnostic_df["predicted_loss"].sum()
)

actual_pure_premium = (
    validation_actual_loss / validation_exposure
)
predicted_pure_premium = (
    validation_predicted_loss / validation_exposure
)

pure_premium_difference_pct = (
    (predicted_pure_premium - actual_pure_premium)
    / actual_pure_premium
    * 100
)

print("\nMATCHED-SUBSET PURE PREMIUM DIAGNOSTIC")
print(f"Actual total loss: {validation_actual_loss:,.2f}")
print(f"Predicted total loss: {validation_predicted_loss:,.2f}")
print(f"Actual pure premium: {actual_pure_premium:,.2f}")
print(f"Predicted pure premium: {predicted_pure_premium:,.2f}")
print(f"Difference (%): {pure_premium_difference_pct:.2f}")
print(
    "Note: this is an in-sample diagnostic using only policies "
    "with complete severity records."
)
print(
    "Missing severity records are concentrated among claiming "
    "policies, so this difference should not be interpreted as "
    "unbiased portfolio calibration."
)


# ASSIGN MATCHED SUBSET TO PURE-PREMIUM RISK DECILES
pure_premium_diagnostic_df["pure_premium_decile"] = pd.qcut(
    pure_premium_diagnostic_df["pure_premium"].rank(method="first"),
    q=N_DECILES,
    labels=False
) + 1

# SUMMARIZE MATCHED-SUBSET LOSS COST BY DECILE
decile_summary = (
    pure_premium_diagnostic_df.groupby(
        "pure_premium_decile",
        observed=True
    )
    .agg(
        policies=("IDpol", "count"),
        exposure=("Exposure", "sum"),
        actual_loss=("actual_total_loss", "sum"),
        predicted_loss=("predicted_loss", "sum")
    )
)

decile_summary["actual_pure_premium"] = (
    decile_summary["actual_loss"]
    / decile_summary["exposure"]
)

decile_summary["predicted_pure_premium"] = (
    decile_summary["predicted_loss"]
    / decile_summary["exposure"]
)

decile_summary["actual_lift"] = (
    decile_summary["actual_pure_premium"]
    / actual_pure_premium
)

decile_summary["predicted_lift"] = (
    decile_summary["predicted_pure_premium"]
    / predicted_pure_premium
)

print("\nMATCHED-SUBSET PURE PREMIUM DECILES")
print(decile_summary)

# PLOT MATCHED-SUBSET ACTUAL VS PREDICTED PURE PREMIUM
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(
    decile_summary.index,
    decile_summary["actual_pure_premium"],
    marker="o",
    label="Actual"
)

ax.plot(
    decile_summary.index,
    decile_summary["predicted_pure_premium"],
    marker="o",
    label="Predicted"
)

ax.set_xlabel("Pure Premium Risk Decile")
ax.set_ylabel("Loss Cost per Exposure Year")
ax.set_title(
    "Matched-Subset Actual vs Predicted Pure Premium by Risk Decile"
)
ax.set_xticks(decile_summary.index)
ax.legend()

fig.tight_layout()

decile_plot_path = (
    OUTPUT_DIR / "pure_premium_decile_diagnostic.png"
)

fig.savefig(
    decile_plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ASSIGN FULL PORTFOLIO TO PURE-PREMIUM RISK DECILES
pricing_df["pure_premium_decile"] = pd.qcut(
    pricing_df["pure_premium"].rank(method="first"),
    q=N_DECILES,
    labels=False
) + 1

# GROUP DECILES INTO BROADER PRICING RISK SEGMENTS
pricing_df["risk_segment"] = pd.cut(
    pricing_df["pure_premium_decile"],
    bins=[0, 4, 7, 10],
    labels=["Low", "Medium", "High"]
)

# SUMMARIZE FULL-PORTFOLIO PRICING RESULTS BY RISK SEGMENT
segment_summary = (
    pricing_df.groupby(
        "risk_segment",
        observed=True
    )
    .agg(
        policies=("IDpol", "count"),
        exposure=("Exposure", "sum"),
        predicted_claims=("predicted_claims", "sum"),
        predicted_loss=("predicted_loss", "sum")
    )
)

segment_summary["predicted_frequency"] = (
    segment_summary["predicted_claims"]
    / segment_summary["exposure"]
)

segment_summary["predicted_severity"] = (
    segment_summary["predicted_loss"]
    / segment_summary["predicted_claims"]
)

segment_summary["pure_premium"] = (
    segment_summary["predicted_loss"]
    / segment_summary["exposure"]
)

print("\nFINAL PRICING RISK SEGMENTS")
print(segment_summary)

# PLOT PURE PREMIUM BY RISK SEGMENT
fig, ax = plt.subplots(figsize=(7, 5))

bars = ax.bar(
    segment_summary.index.astype(str),
    segment_summary["pure_premium"]
)

for bar in bars:
    height = bar.get_height()

    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f"${height:,.0f}",
        ha="center",
        va="bottom"
    )

ax.set_xlabel("Risk Segment")
ax.set_ylabel("Pure Premium per Exposure Year")
ax.set_title("Predicted Pure Premium by Risk Segment")

fig.tight_layout()

segment_plot_path = (
    OUTPUT_DIR / "pure_premium_by_risk_segment.png"
)

fig.savefig(
    segment_plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# DISPLAY POLICY-LEVEL PRICING SAMPLE
print("\nPOLICY-LEVEL PURE PREMIUM SAMPLE")
print(
    pricing_df[
        [
            "IDpol",
            "Exposure",
            "predicted_frequency",
            "predicted_severity",
            "pure_premium",
            "predicted_loss",
            "pure_premium_decile",
            "risk_segment"
        ]
    ].head(10)
)

# SELECT FINAL POLICY-LEVEL PRICING OUTPUT
pricing_output = pricing_df[
    [
        "IDpol",
        "Exposure",
        "Area",
        "VehPower",
        "VehAge",
        "DrivAge",
        "BonusMalus",
        "VehBrand",
        "VehGas",
        "Density",
        "Region",
        "predicted_frequency",
        "predicted_severity",
        "pure_premium",
        "predicted_loss",
        "pure_premium_decile",
        "risk_segment"
    ]
].copy()

# EXPORT FINAL POLICY-LEVEL PRICING RESULTS
output_file = (
    OUTPUT_DIR / "policy_pricing_results.csv"
)

pricing_output.to_csv(
    output_file,
    index=False
)

print("\nFINAL PRICING OUTPUT")
print("Policies exported:", len(pricing_output))
print("Saved as:", output_file.resolve())
print("Decile diagnostic plot:", decile_plot_path.resolve())
print("Risk-segment plot:", segment_plot_path.resolve())





