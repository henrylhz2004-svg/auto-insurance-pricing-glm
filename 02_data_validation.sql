-- ============================================================
-- 02_data_validation.sql
-- Data-quality checks for policy_frequency and claim_severity.
-- ============================================================

USE insurance_project;

-- PORTFOLIO SUMMARY
SELECT
    COUNT(*) AS policy_records,
    COUNT(DISTINCT IDpol) AS unique_policy_ids,
    SUM(Exposure) AS total_exposure,
    SUM(ClaimNb) AS total_reported_claims
FROM policy_frequency;

-- SEVERITY SUMMARY
SELECT
    COUNT(*) AS severity_records,
    COUNT(DISTINCT IDpol) AS claiming_policies,
    SUM(ClaimAmount) AS total_claim_amount,
    AVG(ClaimAmount) AS average_claim_amount,
    MIN(ClaimAmount) AS min_claim_amount,
    MAX(ClaimAmount) AS max_claim_amount
FROM claim_severity;

-- DUPLICATE POLICY IDS
SELECT
    COUNT(*) - COUNT(DISTINCT IDpol) AS duplicate_policy_ids
FROM policy_frequency;

-- CLAIM RECORDS WITH NO MATCHING POLICY
SELECT
    COUNT(*) AS unmatched_claim_records,
    COUNT(DISTINCT c.IDpol) AS unmatched_policy_ids,
    SUM(c.ClaimAmount) AS unmatched_claim_amount
FROM claim_severity AS c
LEFT JOIN policy_frequency AS p
    ON c.IDpol = p.IDpol
WHERE p.IDpol IS NULL;

-- DETAILS FOR UNMATCHED CLAIM POLICY IDS
SELECT
    c.IDpol,
    COUNT(*) AS claim_count,
    SUM(c.ClaimAmount) AS total_claim_amount,
    AVG(c.ClaimAmount) AS average_claim_amount
FROM claim_severity AS c
LEFT JOIN policy_frequency AS p
    ON c.IDpol = p.IDpol
WHERE p.IDpol IS NULL
GROUP BY c.IDpol
ORDER BY claim_count DESC, c.IDpol;

-- COMPARE REPORTED CLAIM COUNTS WITH AVAILABLE SEVERITY RECORDS
WITH claim_counts AS (
    SELECT
        p.IDpol,
        p.ClaimNb,
        COUNT(c.claim_id) AS severity_claim_count
    FROM policy_frequency AS p
    LEFT JOIN claim_severity AS c
        ON p.IDpol = c.IDpol
    GROUP BY
        p.IDpol,
        p.ClaimNb
)
SELECT
    SUM(ClaimNb <> severity_claim_count) AS inconsistent_policies,
    SUM(
        CASE
            WHEN ClaimNb <> severity_claim_count THEN ClaimNb
            ELSE 0
        END
    ) AS reported_claims_in_inconsistent_policies,
    SUM(
        CASE
            WHEN ClaimNb <> severity_claim_count THEN severity_claim_count
            ELSE 0
        END
    ) AS observed_severity_records_in_inconsistent_policies,
    SUM(GREATEST(ClaimNb - severity_claim_count, 0))
        AS missing_severity_records,
    SUM(GREATEST(severity_claim_count - ClaimNb, 0))
        AS excess_severity_records
FROM claim_counts;

-- EXPOSURE RANGE CHECK
SELECT
    MIN(Exposure) AS min_exposure,
    MAX(Exposure) AS max_exposure,
    SUM(Exposure <= 0) AS nonpositive_exposure,
    SUM(Exposure > 1) AS exposure_over_one
FROM policy_frequency;

-- CLAIM-AMOUNT RANGE CHECK
SELECT
    MIN(ClaimAmount) AS min_claim_amount,
    MAX(ClaimAmount) AS max_claim_amount,
    AVG(ClaimAmount) AS average_claim_amount,
    SUM(ClaimAmount <= 0) AS nonpositive_claims
FROM claim_severity;

-- RATING-VARIABLE RANGE CHECK
SELECT
    MIN(DrivAge) AS min_driver_age,
    MAX(DrivAge) AS max_driver_age,
    MIN(VehAge) AS min_vehicle_age,
    MAX(VehAge) AS max_vehicle_age,
    MIN(VehPower) AS min_vehicle_power,
    MAX(VehPower) AS max_vehicle_power,
    MIN(BonusMalus) AS min_bonus_malus,
    MAX(BonusMalus) AS max_bonus_malus,
    MIN(Density) AS min_density,
    MAX(Density) AS max_density
FROM policy_frequency;
