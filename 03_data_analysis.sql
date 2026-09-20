-- ============================================================
-- 03_data_analysis.sql
-- Exploratory claim-frequency analysis by major rating variables.
-- Group definitions match the Python modeling pipeline.
-- ============================================================

USE insurance_project;

-- OVERALL PORTFOLIO FREQUENCY
SELECT
    COUNT(*) AS policy_records,
    SUM(Exposure) AS exposure,
    SUM(ClaimNb) AS claims,
    SUM(ClaimNb) / SUM(Exposure) AS claim_frequency
FROM policy_frequency;


-- DRIVER AGE
SELECT
    MIN(DrivAge) AS min_driver_age,
    MAX(DrivAge) AS max_driver_age,
    COUNT(DISTINCT DrivAge) AS distinct_driver_ages
FROM policy_frequency;

WITH overall AS (
    SELECT
        SUM(ClaimNb) / SUM(Exposure) AS overall_frequency
    FROM policy_frequency
)
SELECT
    CASE
        WHEN DrivAge <= 24 THEN '18-24'
        WHEN DrivAge <= 34 THEN '25-34'
        WHEN DrivAge <= 49 THEN '35-49'
        WHEN DrivAge <= 64 THEN '50-64'
        ELSE '65+'
    END AS driver_age_group,
    COUNT(*) AS policy_records,
    SUM(Exposure) AS exposure,
    SUM(ClaimNb) AS claims,
    SUM(ClaimNb) / SUM(Exposure) AS claim_frequency,
    (
        SUM(ClaimNb) / SUM(Exposure)
    ) / overall.overall_frequency AS frequency_relativity
FROM policy_frequency
CROSS JOIN overall
GROUP BY
    driver_age_group,
    overall.overall_frequency
ORDER BY MIN(DrivAge);


-- BONUS-MALUS
SELECT
    MIN(BonusMalus) AS min_bonus_malus,
    MAX(BonusMalus) AS max_bonus_malus,
    COUNT(DISTINCT BonusMalus) AS distinct_bonus_malus_values
FROM policy_frequency;

WITH overall AS (
    SELECT
        SUM(ClaimNb) / SUM(Exposure) AS overall_frequency
    FROM policy_frequency
)
SELECT
    CASE
        WHEN BonusMalus = 50 THEN '50'
        WHEN BonusMalus <= 75 THEN '51-75'
        WHEN BonusMalus <= 100 THEN '76-100'
        WHEN BonusMalus <= 125 THEN '101-125'
        ELSE '126+'
    END AS bonus_malus_group,
    COUNT(*) AS policy_records,
    SUM(Exposure) AS exposure,
    SUM(ClaimNb) AS claims,
    SUM(ClaimNb) / SUM(Exposure) AS claim_frequency,
    (
        SUM(ClaimNb) / SUM(Exposure)
    ) / overall.overall_frequency AS frequency_relativity
FROM policy_frequency
CROSS JOIN overall
GROUP BY
    bonus_malus_group,
    overall.overall_frequency
ORDER BY MIN(BonusMalus);


-- VEHICLE AGE
SELECT
    MIN(VehAge) AS min_vehicle_age,
    MAX(VehAge) AS max_vehicle_age,
    COUNT(DISTINCT VehAge) AS distinct_vehicle_ages
FROM policy_frequency;

WITH overall AS (
    SELECT
        SUM(ClaimNb) / SUM(Exposure) AS overall_frequency
    FROM policy_frequency
)
SELECT
    CASE
        WHEN VehAge = 0 THEN '0'
        WHEN VehAge <= 5 THEN '1-5'
        WHEN VehAge <= 10 THEN '6-10'
        WHEN VehAge <= 15 THEN '11-15'
        WHEN VehAge <= 20 THEN '16-20'
        ELSE '21+'
    END AS vehicle_age_group,
    COUNT(*) AS policy_records,
    SUM(Exposure) AS exposure,
    SUM(ClaimNb) AS claims,
    SUM(ClaimNb) / SUM(Exposure) AS claim_frequency,
    (
        SUM(ClaimNb) / SUM(Exposure)
    ) / overall.overall_frequency AS frequency_relativity
FROM policy_frequency
CROSS JOIN overall
GROUP BY
    vehicle_age_group,
    overall.overall_frequency
ORDER BY MIN(VehAge);


-- VEHICLE POWER
SELECT
    MIN(VehPower) AS min_vehicle_power,
    MAX(VehPower) AS max_vehicle_power,
    COUNT(DISTINCT VehPower) AS distinct_vehicle_power_values
FROM policy_frequency;

WITH overall AS (
    SELECT
        SUM(ClaimNb) / SUM(Exposure) AS overall_frequency
    FROM policy_frequency
)
SELECT
    VehPower AS vehicle_power,
    COUNT(*) AS policy_records,
    SUM(Exposure) AS exposure,
    SUM(ClaimNb) AS claims,
    SUM(ClaimNb) / SUM(Exposure) AS claim_frequency,
    (
        SUM(ClaimNb) / SUM(Exposure)
    ) / overall.overall_frequency AS frequency_relativity
FROM policy_frequency
CROSS JOIN overall
GROUP BY
    VehPower,
    overall.overall_frequency
ORDER BY VehPower;


-- AREA
WITH overall AS (
    SELECT
        SUM(ClaimNb) / SUM(Exposure) AS overall_frequency
    FROM policy_frequency
)
SELECT
    Area,
    COUNT(*) AS policy_records,
    SUM(Exposure) AS exposure,
    SUM(ClaimNb) AS claims,
    SUM(ClaimNb) / SUM(Exposure) AS claim_frequency,
    (
        SUM(ClaimNb) / SUM(Exposure)
    ) / overall.overall_frequency AS frequency_relativity
FROM policy_frequency
CROSS JOIN overall
GROUP BY
    Area,
    overall.overall_frequency
ORDER BY Area;


-- DENSITY
SELECT
    MIN(Density) AS min_density,
    MAX(Density) AS max_density,
    COUNT(DISTINCT Density) AS distinct_density_values
FROM policy_frequency;

SELECT
    Area,
    COUNT(*) AS policy_records,
    MIN(Density) AS min_density,
    MAX(Density) AS max_density,
    AVG(Density) AS average_density
FROM policy_frequency
GROUP BY Area
ORDER BY Area;


-- VEHICLE FUEL TYPE
WITH overall AS (
    SELECT
        SUM(ClaimNb) / SUM(Exposure) AS overall_frequency
    FROM policy_frequency
)
SELECT
    VehGas,
    COUNT(*) AS policy_records,
    SUM(Exposure) AS exposure,
    SUM(ClaimNb) AS claims,
    SUM(ClaimNb) / SUM(Exposure) AS claim_frequency,
    (
        SUM(ClaimNb) / SUM(Exposure)
    ) / overall.overall_frequency AS frequency_relativity
FROM policy_frequency
CROSS JOIN overall
GROUP BY
    VehGas,
    overall.overall_frequency
ORDER BY VehGas;


-- REGION
SELECT
    COUNT(DISTINCT Region) AS distinct_regions
FROM policy_frequency;

WITH overall AS (
    SELECT
        SUM(ClaimNb) / SUM(Exposure) AS overall_frequency
    FROM policy_frequency
)
SELECT
    Region,
    COUNT(*) AS policy_records,
    SUM(Exposure) AS exposure,
    SUM(ClaimNb) AS claims,
    SUM(ClaimNb) / SUM(Exposure) AS claim_frequency,
    (
        SUM(ClaimNb) / SUM(Exposure)
    ) / overall.overall_frequency AS frequency_relativity
FROM policy_frequency
CROSS JOIN overall
GROUP BY
    Region,
    overall.overall_frequency
ORDER BY claim_frequency DESC;


-- VEHICLE BRAND
SELECT
    COUNT(DISTINCT VehBrand) AS distinct_vehicle_brands
FROM policy_frequency;

WITH overall AS (
    SELECT
        SUM(ClaimNb) / SUM(Exposure) AS overall_frequency
    FROM policy_frequency
)
SELECT
    VehBrand AS vehicle_brand,
    COUNT(*) AS policy_records,
    SUM(Exposure) AS exposure,
    SUM(ClaimNb) AS claims,
    SUM(ClaimNb) / SUM(Exposure) AS claim_frequency,
    (
        SUM(ClaimNb) / SUM(Exposure)
    ) / overall.overall_frequency AS frequency_relativity
FROM policy_frequency
CROSS JOIN overall
GROUP BY
    VehBrand,
    overall.overall_frequency
ORDER BY claim_frequency DESC;
