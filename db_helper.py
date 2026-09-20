import os
from functools import lru_cache

import pandas as pd
from sqlalchemy import URL, create_engine

# DATABASE CONFIGURATION
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "insurance_project")

@lru_cache(maxsize=1)
def get_engine():
    """Create and cache a SQLAlchemy engine for the insurance database."""
    password = os.getenv("MYSQL_PASSWORD")

    if not password:
        raise RuntimeError(
            "MYSQL_PASSWORD environment variable is not set."
        )

    url = URL.create(
        drivername="mysql+mysqlconnector",
        username=MYSQL_USER,
        password=password,
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        database=MYSQL_DATABASE
    )

    return create_engine(
        url,
        pool_pre_ping=True
    )

def load_policy_data():
    """Load the policy-level frequency dataset from MySQL."""
    query = """
        SELECT *
        FROM policy_frequency;
    """

    return pd.read_sql(
        query,
        get_engine()
    )

def load_claim_data():
    """Load the claim-level severity dataset from MySQL."""
    query = """
        SELECT *
        FROM claim_severity;
    """

    return pd.read_sql(
        query,
        get_engine()
    )






