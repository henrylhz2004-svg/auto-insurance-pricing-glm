import pandas as pd
import matplotlib.pyplot as plt

from db_helper import load_policy_data

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_rows", None)

# LOAD POLICY DATA
df = load_policy_data()


# CALCULATE OVERALL PORTFOLIO CLAIM FREQUENCY
total_claims = df["ClaimNb"].sum()
total_exposure = df["Exposure"].sum()
overall_frequency = total_claims / total_exposure

print("Overall claim frequency:", overall_frequency)


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


# BUILD A FREQUENCY SUMMARY FOR ANY CATEGORICAL VARIABLE
def build_frequency_summary(data, group_column):
    summary = (
        data.groupby(group_column, observed=True)
        .agg(
            policy_records=("IDpol", "count"),
            exposure=("Exposure", "sum"),
            claims=("ClaimNb", "sum")
        )
    )

    summary["claim_frequency"] = (
        summary["claims"] / summary["exposure"]
    )

    summary["frequency_relativity"] = (
        summary["claim_frequency"] / overall_frequency
    )

    return summary


# PLOT AND SAVE CLAIM FREQUENCY BY SEGMENT
def plot_frequency_summary(summary, x_label, title, filename):
    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(
        summary.index.astype(str),
        summary["claim_frequency"]
    )

    ax.axhline(
        y=overall_frequency,
        linestyle="--",
        label="Overall portfolio frequency"
    )

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.002,
            f"{height:.3f}",
            ha="center",
            va="bottom"
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel("Claims per Exposure Year")
    ax.set_title(title)
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


# DRIVER AGE ANALYSIS
driver_age_summary = build_frequency_summary(
    df,
    "driver_age_group"
)

print("\nDRIVER AGE SUMMARY")
print(driver_age_summary)

plot_frequency_summary(
    driver_age_summary,
    x_label="Driver Age Group",
    title="Claim Frequency by Driver Age Group",
    filename="driver_age_frequency.png"
)


# BONUS-MALUS ANALYSIS
bonus_malus_summary = build_frequency_summary(
    df,
    "bonus_malus_group"
)

print("\nBONUS-MALUS SUMMARY")
print(bonus_malus_summary)

plot_frequency_summary(
    bonus_malus_summary,
    x_label="Bonus-Malus Group",
    title="Claim Frequency by Bonus-Malus Group",
    filename="bonus_malus_frequency.png"
)


# VEHICLE AGE ANALYSIS
vehicle_age_summary = build_frequency_summary(
    df,
    "vehicle_age_group"
)

print("\nVEHICLE AGE SUMMARY")
print(vehicle_age_summary)

plot_frequency_summary(
    vehicle_age_summary,
    x_label="Vehicle Age Group",
    title="Claim Frequency by Vehicle Age Group",
    filename="vehicle_age_frequency.png"
)


# AREA ANALYSIS
area_summary = build_frequency_summary(
    df,
    "Area"
)

print("\nAREA SUMMARY")
print(area_summary)

plot_frequency_summary(
    area_summary,
    x_label="Area",
    title="Claim Frequency by Area",
    filename="area_frequency.png"
)


# VEHICLE BRAND ANALYSIS
vehicle_brand_summary = build_frequency_summary(
    df,
    "VehBrand"
).sort_values(
    "claim_frequency",
    ascending=False
)

print("\nVEHICLE BRAND SUMMARY")
print(vehicle_brand_summary)

plot_frequency_summary(
    vehicle_brand_summary,
    x_label="Vehicle Brand",
    title="Claim Frequency by Vehicle Brand",
    filename="vehicle_brand_frequency.png"
)






