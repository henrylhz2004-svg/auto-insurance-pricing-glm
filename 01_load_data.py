import pandas as pd

from db_helper import load_policy_data, load_claim_data


# CONFIGURE PANDAS DISPLAY
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)


def summarize_dataset(name, data):
    """Print a compact structural and data-quality summary."""
    print(f"\n{name}")
    print("-" * len(name))

    print("Shape:", data.shape)

    print("\nFirst 5 rows:")
    print(data.head())

    print("\nData types:")
    print(data.dtypes)

    print("\nMissing values:")
    print(data.isnull().sum())


def main():
    # LOAD SOURCE DATA
    policy_df = load_policy_data()
    claims_df = load_claim_data()

    # REVIEW POLICY DATA
    summarize_dataset(
        "POLICY FREQUENCY DATA",
        policy_df
    )

    # REVIEW CLAIM DATA
    summarize_dataset(
        "CLAIM SEVERITY DATA",
        claims_df
    )

    # BASIC KEY CHECKS
    print("\nKEY CHECKS")
    print(
        "Duplicate policy IDs in policy data:",
        policy_df["IDpol"].duplicated().sum()
    )
    print(
        "Unique claiming policies in severity data:",
        claims_df["IDpol"].nunique()
    )


if __name__ == "__main__":
    main()








