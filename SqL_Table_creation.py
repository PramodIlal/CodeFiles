-- =========================================================
-- STEP 1: CREATE DATABASE
-- =========================================================
CREATE DATABASE IF NOT EXISTS etl_db;

-- =========================================================
-- STEP 2: USE DATABASE
-- =========================================================
USE etl_db;

-- =========================================================
-- STEP 3: DROP TABLES IF THEY ALREADY EXIST (OPTIONAL)
-- =========================================================
DROP TABLE IF EXISTS source_orders;
DROP TABLE IF EXISTS target_orders;

-- =========================================================
-- STEP 4: CREATE SOURCE TABLE
-- =========================================================
CREATE TABLE source_orders (
    order_id INT PRIMARY KEY,
    customer_name VARCHAR(100),
    amount DECIMAL(10,2),
    status VARCHAR(20),
    created_at DATETIME
);

-- =========================================================
-- STEP 5: CREATE TARGET TABLE
-- =========================================================
CREATE TABLE target_orders (
    order_id INT PRIMARY KEY,
    customer_name VARCHAR(100),
    amount DECIMAL(10,2),
    status VARCHAR(20),
    created_at DATETIME
);

-- =========================================================
-- STEP 6: INSERT SAMPLE DATA INTO SOURCE TABLE
-- =========================================================
INSERT INTO source_orders (order_id, customer_name, amount, status, created_at) VALUES
(1, 'Alice',   1200.50, 'ACTIVE',   NOW()),
(2, 'Bob',      850.00, 'ACTIVE',   NOW()),
(3, 'Charlie', 1500.75, 'INACTIVE', NOW());

-- =========================================================
-- STEP 7: INSERT SAMPLE DATA INTO TARGET TABLE
-- =========================================================
INSERT INTO target_orders (order_id, customer_name, amount, status, created_at) VALUES
(1, 'Alice', 1200.50, 'ACTIVE', NOW()),
(2, 'Bob',    900.00, 'ACTIVE', NOW()),
(4, 'David',  500.00, 'ACTIVE', NOW());

-- =========================================================
-- STEP 8: CREATE DEDICATED USER FOR ETL PROJECT
-- (For local machine use localhost)
-- =========================================================
CREATE USER IF NOT EXISTS 'etl_user'@'localhost' IDENTIFIED BY 'StrongPassword123!';

-- =========================================================
-- STEP 9: GRANT ACCESS TO ETL USER
-- =========================================================
GRANT ALL PRIVILEGES ON etl_db.* TO 'etl_user'@'localhost';

-- =========================================================
-- STEP 10: APPLY CHANGES
-- =========================================================
FLUSH PRIVILEGES;

-- =========================================================
-- STEP 11: VERIFY DATA
-- =========================================================
SELECT * FROM source_orders;
SELECT * FROM target_orders;
