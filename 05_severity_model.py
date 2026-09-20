import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from scipy.stats import chi2
from db_helper import load_policy_data, load_claim_data

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_rows", None)

# LOAD SOURCE DATA
policy_df = load_policy_data()
claims_df = load_claim_data()

# VALIDATE MODELING INPUTS
print("\nCLAIM AMOUNT CHECK")
print("Minimum claim amount:", claims_df["ClaimAmount"].min())
print("Maximum claim amount:", claims_df["ClaimAmount"].max())
print(
    "Zero or negative claims:",
    (claims_df["ClaimAmount"] <= 0).sum()
)

assert (claims_df["ClaimAmount"] > 0).all(), (
    "Gamma severity modeling requires strictly positive claim amounts."
)

assert (policy_df["Density"] > 0).all(), (
    "Density contains zero or negative values."
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
        "Density",
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

    data["log_density"] = np.log(data["Density"])

    return data

# AGGREGATE CLAIMS AND BUILD THE MAIN SEVERITY DATASET
def build_severity_data(claim_data, validate_claim_counts=True):
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

severity_df, mismatch_count = build_severity_data(
    claims_df,
    validate_claim_counts=True
)

print("\nSEVERITY DATA QUALITY")
print("Policies with claim-count mismatch:", mismatch_count)
print("Severity policies used:", len(severity_df))

# DEFINE SEVERITY MODEL SPECIFICATIONS
model_1_formula = """
    average_severity ~
    C(driver_age_group, Treatment(reference="35-49"))
    + C(bonus_malus_group, Treatment(reference="50"))
"""

model_2_formula = """
    average_severity ~
    C(driver_age_group, Treatment(reference="35-49"))
    + C(bonus_malus_group, Treatment(reference="50"))
    + C(vehicle_age_group, Treatment(reference="6-10"))
    + C(VehPower)
    + C(VehBrand)
    + C(VehGas)
"""

density_formula = """
    average_severity ~
    C(driver_age_group, Treatment(reference="35-49"))
    + C(bonus_malus_group, Treatment(reference="50"))
    + C(vehicle_age_group, Treatment(reference="6-10"))
    + C(VehPower)
    + C(VehBrand)
    + C(VehGas)
    + log_density
    + C(Region)
"""

area_formula = """
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

area_no_region_formula = """
    average_severity ~
    C(driver_age_group, Treatment(reference="35-49"))
    + C(bonus_malus_group, Treatment(reference="50"))
    + C(vehicle_age_group, Treatment(reference="6-10"))
    + C(VehPower)
    + C(VehBrand)
    + C(VehGas)
    + C(Area)
"""

# FIT A GAMMA GLM WITH CLAIM-COUNT PRECISION WEIGHTS
def fit_gamma(formula, data):
    return smf.glm(
        formula=formula,
        data=data,
        family=sm.families.Gamma(
            link=sm.families.links.Log()
        ),
        var_weights=data["severity_claim_count"]
    ).fit()

# FIT BASELINE GAMMA SEVERITY MODEL
severity_model_0 = fit_gamma(
    "average_severity ~ 1",
    severity_df
)

baseline_severity = np.exp(
    severity_model_0.params["Intercept"]
)

print("\nBASELINE SEVERITY MODEL")
print("Baseline severity:", baseline_severity)

# FIT SEVERITY MODEL 1: DRIVER AGE + BONUS MALUS
severity_model_1 = fit_gamma(
    model_1_formula,
    severity_df
)

# FIT SEVERITY MODEL 2: ADD VEHICLE CHARACTERISTICS
severity_model_2 = fit_gamma(
    model_2_formula,
    severity_df
)

# TEST VEHICLE CHARACTERISTICS AS A BLOCK
vehicle_lr_stat = 2 * (
    severity_model_2.llf
    - severity_model_1.llf
)

vehicle_df_diff = (
    severity_model_2.df_model
    - severity_model_1.df_model
)

vehicle_lr_pvalue = chi2.sf(
    vehicle_lr_stat,
    vehicle_df_diff
)

print("\nLIKELIHOOD RATIO TEST FOR VEHICLE CHARACTERISTICS")
print("LR statistic:", vehicle_lr_stat)
print("Degrees of freedom:", vehicle_df_diff)
print(
    "p-value:",
    "< 0.001" if vehicle_lr_pvalue < 0.001
    else f"{vehicle_lr_pvalue:.3f}"
)

# FIT GEOGRAPHIC MODEL USING LOG DENSITY
severity_model_3_density = fit_gamma(
    density_formula,
    severity_df
)

# TEST GEOGRAPHIC CHARACTERISTICS AS A BLOCK
geo_lr_stat = 2 * (
    severity_model_3_density.llf
    - severity_model_2.llf
)

geo_df_diff = (
    severity_model_3_density.df_model
    - severity_model_2.df_model
)

geo_lr_pvalue = chi2.sf(
    geo_lr_stat,
    geo_df_diff
)

print("\nLIKELIHOOD RATIO TEST FOR GEOGRAPHIC CHARACTERISTICS")
print("LR statistic:", geo_lr_stat)
print("Degrees of freedom:", geo_df_diff)
print(
    "p-value:",
    "< 0.001" if geo_lr_pvalue < 0.001
    else f"{geo_lr_pvalue:.3f}"
)

# FIT ALTERNATIVE GEOGRAPHIC MODEL USING AREA
severity_model_3_area = fit_gamma(
    area_formula,
    severity_df
)

# SUMMARIZE SEVERITY MODEL DEVELOPMENT
model_comparison = pd.DataFrame(
    {
        "AIC": [
            severity_model_0.aic,
            severity_model_1.aic,
            severity_model_2.aic,
            severity_model_3_area.aic
        ],
        "Deviance": [
            severity_model_0.deviance,
            severity_model_1.deviance,
            severity_model_2.deviance,
            severity_model_3_area.deviance
        ]
    },
    index=[
        "Model 0: Baseline",
        "Model 1: Driver + BonusMalus",
        "Model 2: + Vehicle",
        "Model 3: + Area + Region"
    ]
)

print("\nSEVERITY MODEL DEVELOPMENT")
print(model_comparison)

# COMPARE AREA VS LOG DENSITY
print("\nAREA VS DENSITY FOR SEVERITY")
print("Area model AIC:", severity_model_3_area.aic)
print("Density model AIC:", severity_model_3_density.aic)
print("Area model Deviance:", severity_model_3_area.deviance)
print("Density model Deviance:", severity_model_3_density.deviance)

# FIT REDUCED AREA MODEL WITHOUT REGION
severity_model_area_no_region = fit_gamma(
    area_no_region_formula,
    severity_df
)

# TEST REGION CONDITIONAL ON AREA
region_lr_stat = 2 * (
    severity_model_3_area.llf
    - severity_model_area_no_region.llf
)

region_df_diff = (
    severity_model_3_area.df_model
    - severity_model_area_no_region.df_model
)

region_lr_pvalue = chi2.sf(
    region_lr_stat,
    region_df_diff
)

print("\nREGION TEST WITH AREA")
print("Without Region AIC:", severity_model_area_no_region.aic)
print("With Region AIC:", severity_model_3_area.aic)
print("LR statistic:", region_lr_stat)
print("Degrees of freedom:", region_df_diff)
print(
    "p-value:",
    "< 0.001" if region_lr_pvalue < 0.001
    else f"{region_lr_pvalue:.3f}"
)

# PREPARE LARGE-LOSS SENSITIVITY DATASET
# First restrict to policies that passed the original claim-count validation.
valid_policy_ids = severity_df["IDpol"]

claims_df_sensitivity = claims_df[
    claims_df["IDpol"].isin(valid_policy_ids)
].copy()

largest_claim_idx = claims_df_sensitivity["ClaimAmount"].idxmax()

claims_df_sensitivity = claims_df_sensitivity.drop(
    index=largest_claim_idx
).copy()

# Claim-count validation is intentionally disabled here because one observed
# claim has been removed for the sensitivity test.
severity_df_sensitivity, _ = build_severity_data(
    claims_df_sensitivity,
    validate_claim_counts=False
)

# FIT FINAL LARGE-LOSS SENSITIVITY MODEL
severity_final_sensitivity = fit_gamma(
    area_formula,
    severity_df_sensitivity
)

# CALCULATE FINAL MODEL RELATIVITIES
final_relativities = np.exp(
    severity_model_3_area.params.drop(labels="Intercept")
)

final_sensitivity_relativities = np.exp(
    severity_final_sensitivity.params.drop(labels="Intercept")
)

age_term = (
    'C(driver_age_group, Treatment(reference="35-49"))[T.18-24]'
)

# REPORT FINAL SEVERITY MODEL
print("\nFINAL SEVERITY MODEL")
print("AIC:", severity_model_3_area.aic)
print("Deviance:", severity_model_3_area.deviance)

print("\nFINAL MODEL LARGE-LOSS SENSITIVITY")
print("Original 18-24 relativity:", final_relativities[age_term])
print(
    "Without largest claim:",
    final_sensitivity_relativities[age_term]
)

print("\nFINAL SEVERITY MODEL COEFFICIENTS")
print(severity_model_3_area.summary().tables[1])

print("\nFINAL SEVERITY RELATIVITIES")
print(final_relativities)





