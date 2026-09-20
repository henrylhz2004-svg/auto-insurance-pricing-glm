import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from scipy.stats import chi2
from db_helper import load_policy_data

# LOAD POLICY DATA
df = load_policy_data()

# VALIDATE POSITIVE VARIABLES USED IN LOG TRANSFORMS
assert (df["Exposure"] > 0).all(), "Exposure contains zero or negative values."
assert (df["Density"] > 0).all(), "Density contains zero or negative values."

# CREATE MODELING VARIABLES
df["log_exposure"] = np.log(df["Exposure"])
df["log_density"] = np.log(df["Density"])

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

# DEFINE MODEL SPECIFICATIONS
formula_model_1 = """
    ClaimNb ~
    C(driver_age_group, Treatment(reference="35-49"))
    + C(bonus_malus_group, Treatment(reference="50"))
"""

formula_model_2 = """
    ClaimNb ~
    C(driver_age_group, Treatment(reference="35-49"))
    + C(bonus_malus_group, Treatment(reference="50"))
    + C(vehicle_age_group, Treatment(reference="6-10"))
    + C(VehPower)
    + C(VehBrand)
    + C(VehGas)
"""

formula_model_3_area = """
    ClaimNb ~
    C(driver_age_group, Treatment(reference="35-49"))
    + C(bonus_malus_group, Treatment(reference="50"))
    + C(vehicle_age_group, Treatment(reference="6-10"))
    + C(VehPower)
    + C(VehBrand)
    + C(VehGas)
    + C(Area)
    + C(Region)
"""

formula_model_3_density = """
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

formula_density_no_region = """
    ClaimNb ~
    C(driver_age_group, Treatment(reference="35-49"))
    + C(bonus_malus_group, Treatment(reference="50"))
    + C(vehicle_age_group, Treatment(reference="6-10"))
    + C(VehPower)
    + C(VehBrand)
    + C(VehGas)
    + log_density
"""

# HELPER FUNCTIONS
def fit_poisson(formula):
    return smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Poisson(),
        offset=df["log_exposure"]
    ).fit()

def dispersion(model):
    return model.pearson_chi2 / model.df_resid

def format_p_value(p_value):
    if p_value == 0.0:
        return "< 1e-300"
    if p_value < 0.001:
        return f"{p_value:.3e}"
    return f"{p_value:.4f}"

# FIT BASELINE POISSON MODEL
model_0 = fit_poisson("ClaimNb ~ 1")
baseline_frequency = np.exp(model_0.params["Intercept"])

# FIT POISSON MODEL 1: DRIVER AGE + BONUS MALUS
model_1 = fit_poisson(formula_model_1)

# FIT POISSON MODEL 2: ADD VEHICLE CHARACTERISTICS
model_2 = fit_poisson(formula_model_2)

# FIT POISSON MODEL 3A: ADD AREA + REGION
model_3_area = fit_poisson(formula_model_3_area)

# FIT POISSON MODEL 3D: ADD LOG DENSITY + REGION
model_3_density = fit_poisson(formula_model_3_density)

# REPORT POISSON MODEL DEVELOPMENT
poisson_models = {
    "Model 0": model_0,
    "Model 1": model_1,
    "Model 2": model_2,
    "Model 3D": model_3_density,
}

print("\nBASELINE FREQUENCY MODEL")
print("Baseline frequency:", baseline_frequency)

print("\nPOISSON MODEL DEVELOPMENT")
for model_name, model in poisson_models.items():
    print(
        f"{model_name}: "
        f"AIC={model.aic:.3f}, "
        f"Deviance={model.deviance:.3f}, "
        f"Dispersion={dispersion(model):.3f}"
    )

# COMPARE AREA VS LOG DENSITY FOR GEOGRAPHIC SPECIFICATION
print("\nAREA VS DENSITY FOR FREQUENCY")
print("Area model AIC:", model_3_area.aic)
print("Density model AIC:", model_3_density.aic)
print("Area model Deviance:", model_3_area.deviance)
print("Density model Deviance:", model_3_density.deviance)
print("Area model dispersion:", dispersion(model_3_area))
print("Density model dispersion:", dispersion(model_3_density))

# FIT REDUCED DENSITY MODEL WITHOUT REGION
model_density_no_region = fit_poisson(formula_density_no_region)

# TEST REGION CONDITIONAL ON LOG DENSITY
region_lr_stat = 2 * (
    model_3_density.llf
    - model_density_no_region.llf
)

region_df_diff = (
    model_3_density.df_model
    - model_density_no_region.df_model
)

region_lr_pvalue = chi2.sf(
    region_lr_stat,
    region_df_diff
)

print("\nREGION TEST WITH LOG DENSITY")
print("Without Region AIC:", model_density_no_region.aic)
print("With Region AIC:", model_3_density.aic)
print("Without Region Deviance:", model_density_no_region.deviance)
print("With Region Deviance:", model_3_density.deviance)
print("LR statistic:", region_lr_stat)
print("Degrees of freedom:", region_df_diff)
print("p-value:", format_p_value(region_lr_pvalue))

# FIT NEGATIVE BINOMIAL MODEL TO ADDRESS OVERDISPERSION
nb_model = smf.negativebinomial(
    formula=formula_model_3_density,
    data=df,
    exposure=df["Exposure"]
)

# Use Poisson estimates as stable starting values and initialize alpha at 1.0.
start_params = np.append(
    model_3_density.params.to_numpy(),
    1.0
)

model_nb = nb_model.fit(
    start_params=start_params,
    method="bfgs",
    maxiter=200,
    disp=False
)

# COMPARE POISSON AND NEGATIVE BINOMIAL MODELS
print("\nPOISSON VS NEGATIVE BINOMIAL")
print("Poisson dispersion:", dispersion(model_3_density))
print("Poisson AIC:", model_3_density.aic)
print("Negative Binomial AIC:", model_nb.aic)
print("AIC improvement:", model_3_density.aic - model_nb.aic)
print("Converged:", model_nb.mle_retvals["converged"])
print("Alpha:", model_nb.params["alpha"])


# CALCULATE FINAL NEGATIVE BINOMIAL RELATIVITIES
final_nb_relativities = np.exp(
    model_nb.params.drop(labels=["Intercept", "alpha"], errors="ignore")
)

# REPORT FINAL FREQUENCY MODEL RESULTS
print("\nFINAL NEGATIVE BINOMIAL MODEL COEFFICIENTS")
print(model_nb.summary().tables[1])

print("\nFINAL FREQUENCY RELATIVITIES")
print(final_nb_relativities)






