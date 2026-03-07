-- Usage:
-- sqlite3 tou.db ".read query_examples.sql"
-- You can change 'GD' and dates as needed.

PRAGMA foreign_keys = ON;

-- 1) Basic overview
SELECT 'tables' AS section, name
FROM sqlite_master
WHERE type = 'table'
ORDER BY name;

SELECT 'province_count' AS section, COUNT(*) AS cnt
FROM province;

SELECT 'ts_value_total' AS section, COUNT(*) AS cnt
FROM ts_value;

-- 2) Province list
SELECT id, code, name, timezone, created_at
FROM province
ORDER BY id;

-- 3) TOU rule and segments for Guangdong
SELECT r.id AS rule_id, p.code, r.rule_name, r.effective_from, r.effective_to, r.resolution_min
FROM tou_rule r
JOIN province p ON p.id = r.province_id
WHERE p.code = 'GD'
ORDER BY r.effective_from;

SELECT p.code, r.rule_name, s.period, s.start_minute, s.end_minute
FROM tou_rule_segment s
JOIN tou_rule r ON r.id = s.rule_id
JOIN province p ON p.id = r.province_id
WHERE p.code = 'GD'
ORDER BY r.effective_from, s.start_minute;

-- 4) Data distribution by metric/period/source
SELECT metric, COUNT(*) AS cnt
FROM ts_value
GROUP BY metric
ORDER BY metric;

SELECT period, COUNT(*) AS cnt
FROM ts_value
GROUP BY period
ORDER BY period;

SELECT source, COUNT(*) AS cnt
FROM ts_value
GROUP BY source
ORDER BY source;

-- 5) Time range in table
SELECT MIN(ts) AS min_ts, MAX(ts) AS max_ts, COUNT(DISTINCT ts) AS distinct_ts
FROM ts_value;

-- 6) Daily average prices by TOU period (example day)
SELECT period, AVG(value) AS avg_price
FROM ts_value
WHERE province_id = (SELECT id FROM province WHERE code = 'GD')
  AND metric = 'real_time_price'
  AND date(ts) = '2026-01-15'
GROUP BY period
ORDER BY period;

-- 7) Monthly average day-ahead price by TOU period
SELECT strftime('%Y-%m', ts) AS ym, period, AVG(value) AS avg_val
FROM ts_value
WHERE province_id = (SELECT id FROM province WHERE code = 'GD')
  AND metric = 'day_ahead_price'
GROUP BY ym, period
ORDER BY ym, period;

-- 8) Intraday profile (96 points) average by HH:MM
SELECT strftime('%H:%M', ts) AS hhmm, AVG(value) AS avg_rt_price
FROM ts_value
WHERE province_id = (SELECT id FROM province WHERE code = 'GD')
  AND metric = 'real_time_price'
GROUP BY hhmm
ORDER BY hhmm;

-- 9) Example raw slice
SELECT ts, metric, value, period, source
FROM ts_value
WHERE province_id = (SELECT id FROM province WHERE code = 'GD')
  AND ts >= '2026-02-01 00:00:00'
  AND ts <  '2026-02-02 00:00:00'
ORDER BY ts, metric;

