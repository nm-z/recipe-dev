CREATE TEMP TABLE quota_observed AS
SELECT *,
    CASE
        WHEN resets_at = previous_resets_at AND used_percent >= previous_used_percent
        THEN used_percent - previous_used_percent
        ELSE 0
    END AS observed_increment
FROM quota_corpus;

CREATE TEMP TABLE quota_constants AS
WITH mean AS (
    SELECT avg(observed_increment) AS mean_y FROM quota_observed
)
SELECT mean_y,
    sum(CASE WHEN observed_increment > 0
        THEN observed_increment * ln(observed_increment) - observed_increment
        ELSE 0 END) AS deviance_constant,
    sum(2 * CASE
        WHEN observed_increment = 0 THEN mean_y
        ELSE observed_increment * ln(observed_increment / mean_y)
            - (observed_increment - mean_y)
    END) AS baseline_deviance,
    sum(pow(observed_increment - mean_y, 2)) AS baseline_sse
FROM quota_observed CROSS JOIN mean
GROUP BY mean_y;
