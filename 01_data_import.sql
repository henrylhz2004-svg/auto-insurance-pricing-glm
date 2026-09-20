-- ============================================================
-- 01_data_import.sql
-- Load the frequency and severity CSV files into MySQL.
-- Replace the placeholder paths before running.
-- ============================================================

USE insurance_project;

SHOW GLOBAL VARIABLES LIKE 'local_infile';

-- If LOCAL INFILE is disabled, enable it separately with
-- sufficient privileges:
-- SET GLOBAL local_infile = ON;

-- CLEAR EXISTING DATA
TRUNCATE TABLE claim_severity;
TRUNCATE TABLE policy_frequency;

-- LOAD POLICY FREQUENCY DATA
LOAD DATA LOCAL INFILE '/path/to/freMTPL2freq.csv'
INTO TABLE policy_frequency
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS
(
    IDpol,
    ClaimNb,
    Exposure,
    Area,
    VehPower,
    VehAge,
    DrivAge,
    BonusMalus,
    VehBrand,
    VehGas,
    Density,
    Region
);

-- LOAD CLAIM SEVERITY DATA
LOAD DATA LOCAL INFILE '/path/to/freMTPL2sev.csv'
INTO TABLE claim_severity
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS
(
    IDpol,
    ClaimAmount
);

-- VERIFY IMPORTED ROW COUNTS
SELECT COUNT(*) AS policy_rows
FROM policy_frequency;

SELECT COUNT(*) AS severity_rows
FROM claim_severity;
