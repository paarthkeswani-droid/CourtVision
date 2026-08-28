-- Model value relative to the draft market (lower pick numbers are better).
SELECT player, season, draft_pick, ROUND(model_grade, 1) AS grade,
       ROUND(model_grade - (101 - draft_pick), 1) AS value_gap
FROM prospects WHERE draft_pick IS NOT NULL
ORDER BY value_gap DESC LIMIT 25;

-- Young, efficient scorers.
SELECT player, season, age, ROUND(ts_pct, 3) AS ts,
       ROUND(pts_per40, 1) AS pts_per40
FROM prospects WHERE age <= 20.5 AND ts_pct >= 0.58
ORDER BY ts_pct DESC;

-- Playmaking profiles.
SELECT player, season, ROUND(ast_tov, 2) AS ast_to_tov,
       ROUND(ast_per40, 1) AS ast_per40
FROM prospects WHERE ast_per40 >= 4
ORDER BY ast_tov DESC;

-- Defensive-event producers.
SELECT player, season, ROUND(stl_per40 + blk_per40, 1) AS events_per40
FROM prospects ORDER BY events_per40 DESC LIMIT 25;

