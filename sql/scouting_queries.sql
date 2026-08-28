-- Load outputs/prospect_rankings.csv into a SQLite table named prospects.

-- Model favorites relative to draft position (smaller draft_pick is better market rank).
SELECT player, season, school, model_grade, draft_pick
FROM prospects
WHERE draft_pick IS NOT NULL
ORDER BY model_grade DESC, draft_pick DESC
LIMIT 20;

-- Young, efficient scoring profiles.
SELECT player, season, age, ts_pct, pts_per40, model_grade
FROM prospects
WHERE age <= 20.5
ORDER BY ts_pct DESC, pts_per40 DESC
LIMIT 20;

-- Two-way event production.
SELECT player, stl_per40, blk_per40, model_grade
FROM prospects
ORDER BY (stl_per40 + blk_per40) DESC
LIMIT 20;

-- Playmaking efficiency.
SELECT player, ast_per40, ast_tov, model_grade
FROM prospects
WHERE ast_per40 >= 3
ORDER BY ast_tov DESC
LIMIT 20;
