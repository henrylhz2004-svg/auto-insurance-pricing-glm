-- ============================================================
-- 00_create_schema.sql
-- Create the MySQL database and tables used by the project.
-- ============================================================

CREATE DATABASE IF NOT EXISTS insurance_project
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_0900_ai_ci;

USE insurance_project;


-- DROP EXISTING TABLES
DROP TABLE IF EXISTS claim_severity;
DROP TABLE IF EXISTS policy_frequency;


-- CREATE POLICY FREQUENCY TABLE
CREATE TABLE policy_frequency (
    IDpol INT NOT NULL,
    ClaimNb INT DEFAULT NULL,
    Exposure DECIMAL(10,8) DEFAULT NULL,
    Area VARCHAR(1) DEFAULT NULL,
    VehPower INT DEFAULT NULL,
    VehAge INT DEFAULT NULL,
    DrivAge INT DEFAULT NULL,
    BonusMalus INT DEFAULT NULL,
    VehBrand VARCHAR(10) DEFAULT NULL,
    VehGas VARCHAR(10) DEFAULT NULL,
    Density INT DEFAULT NULL,
    Region VARCHAR(10) DEFAULT NULL,
    PRIMARY KEY (IDpol)
) ENGINE=InnoDB
  DEFAULT CHARSET=utf8mb4
  COLLATE=utf8mb4_0900_ai_ci;


-- CREATE CLAIM SEVERITY TABLE
CREATE TABLE claim_severity (
    claim_id INT NOT NULL AUTO_INCREMENT,
    IDpol INT DEFAULT NULL,
    ClaimAmount DECIMAL(15,2) DEFAULT NULL,
    PRIMARY KEY (claim_id)
) ENGINE=InnoDB
  DEFAULT CHARSET=utf8mb4
  COLLATE=utf8mb4_0900_ai_ci;


-- VERIFY CREATED TABLES
SHOW TABLES;

DESCRIBE policy_frequency;
DESCRIBE claim_severity;
