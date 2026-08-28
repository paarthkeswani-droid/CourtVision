# CourtVision 🏀

**NBA/NCAA Draft Prospect & Player Development Model**

CourtVision is a portfolio-ready sports analytics project for evaluating basketball prospects with public data. It turns college production, age, size, efficiency, playmaking, rebounding and defensive-event indicators into an explainable prospect model, historical NBA comparisons and scouting-oriented outputs.

> **Decision question:** Which prospects have statistical profiles associated with NBA success, who is undervalued relative to draft position, and which historical players are the closest quantitative comparisons?

## What this project demonstrates

- Python data engineering with `pandas` and `numpy`
- SQL analytics with SQLite
- Feature engineering and leakage-aware modeling
- `scikit-learn` regression, cross-validation and evaluation
- Nearest-neighbor player similarity
- Explainable prospect grades rather than black-box rankings
- Professional visualizations for scouting communication
- Reproducible command-line workflow

## Project structure

```text
CourtVision/
├── data/
│   ├── raw/                 # user/public source inputs (gitignored)
│   └── processed/           # model-ready outputs (gitignored)
├── outputs/                 # rankings, metrics and charts
├── sql/
│   └── scouting_queries.sql
├── src/
│   ├── build_dataset.py
│   ├── features.py
│   ├── model.py
│   └── similarities.py
├── tests/
│   └── test_features.py
├── .gitignore
├── LICENSE
├── Makefile
├── requirements.txt
└── README.md
```

## Data

CourtVision deliberately does **not** commit scraped or redistributed proprietary datasets. The pipeline accepts CSV files assembled from public/statistical sources. Recommended sources include NCAA statistics pages/data exports for college production and NBA public statistics endpoints for professional outcomes. Basketball Reference can be used for manual research, subject to its terms of use.

Place a prospect-season file at `data/raw/college_players.csv` and NBA outcomes at `data/raw/nba_outcomes.csv`.

### Required college columns

`player, season, age, height_in, weight_lb, games, minutes, pts, fga, fta, three_pa, three_p, ft, ast, tov, orb, drb, stl, blk`

Optional columns such as `school`, `conference`, `draft_pick` and `class_year` are retained when present.

### Required outcome columns

`player, nba_minutes, nba_ws, nba_bpm`

The default target is a transparent early-career value index derived from NBA minutes, Win Shares and BPM. You can replace it with a custom outcome in `src/build_dataset.py`.

## Feature engineering

The model uses rate/efficiency features rather than raw box-score totals where possible:

- True shooting percentage
- Assist-to-turnover ratio
- Points, assists, rebounds, steals and blocks per 40 minutes
- Three-point attempt rate
- Free-throw rate
- Age and physical measurements

The goal is not to claim these features are a complete scouting system. Film, role, injury history, competition, team context, measurements and tracking data matter too.

## Modeling

`src/model.py` compares two interpretable baselines:

1. **Elastic Net** — regularized linear model that makes directional relationships easy to inspect.
2. **Random Forest** — nonlinear model capable of learning interactions between age, efficiency, size and production.

The script uses cross-validation on the training set, reports MAE/RMSE/R² on a held-out test set, selects the lower-MAE model, produces percentile-based prospect grades, and writes feature importance/coefficient output.

### Why this design?

A scouting model should be evaluated out of sample and should communicate uncertainty. CourtVision therefore treats the model as a **decision-support tool**, not a replacement for scouting.

## Historical player comparisons

`src/similarities.py` standardizes prospect features and uses cosine similarity to find the closest historical statistical profiles. This gives a recruiter/scout an intuitive answer to: *"What kind of prospect has looked like this before?"*

Similarity is descriptive, not destiny. A 90% statistical match does not imply a 90% chance of the same career.

## SQL scouting layer

Load the model output into SQLite and use `sql/scouting_queries.sql` for examples such as:

- biggest model-vs-draft-position value gaps
- young high-efficiency prospects
- guards/wings with strong assist-to-turnover profiles
- high-event defenders

This makes the project relevant to general analyst roles as well as basketball operations.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python src/build_dataset.py
python src/model.py
python src/similarities.py --player "PLAYER NAME"
```

Or run:

```bash
make all
```

## Outputs

After running the pipeline:

- `data/processed/model_data.csv` — model-ready historical dataset
- `outputs/model_metrics.json` — holdout metrics
- `outputs/prospect_rankings.csv` — predicted value + 0–100 grade
- `outputs/feature_importance.csv` — model interpretation
- `outputs/predicted_vs_actual.png` — evaluation visual
- `outputs/top_prospects.png` — portfolio-ready ranking chart
- `outputs/similar_players.csv` — historical comps for a selected player

## Scouting interpretation

A strong CourtVision report should answer three questions:

1. **What does the model see?** Production, efficiency, age and physical profile.
2. **Why does it matter?** Which features historically associate with the target outcome.
3. **What is missing?** Film/context information that should be investigated before making a personnel decision.

Example scouting language:

> The model views Prospect X as an above-average statistical bet because of age-adjusted scoring efficiency, playmaking and steal production. His closest historical profiles suggest starter-level upside, but the grade should be discounted until role, competition level and defensive film are reviewed.

## Limitations

- Public NCAA data can require manual collection/cleaning and name matching.
- Box-score statistics do not capture spacing, screen navigation, decision speed or many defensive responsibilities.
- Historical NBA outcomes are affected by opportunity and team context.
- Draft selection itself contains information; including `draft_pick` can improve prediction but introduces market consensus into the model. The default feature set excludes it so model-vs-market comparisons remain meaningful.
- Small samples and changing basketball environments can reduce stability.

## Next improvements

- Add conference/strength-of-schedule adjustments
- Separate guard, wing and big models
- Add possession/play-type or tracking features where licensing permits
- Use time-based draft-class validation
- Build a Streamlit scouting dashboard
- Add calibration for outcome tiers (All-Star / starter / rotation / fringe)

## Resume bullets

- **Built CourtVision, an end-to-end basketball scouting analytics pipeline using Python, pandas, SQL and scikit-learn to engineer prospect features, predict early NBA value and identify historical player comparisons.**
- **Evaluated Elastic Net and Random Forest models with cross-validation and held-out MAE/RMSE/R², translating model outputs into explainable 0–100 prospect grades and scouting visuals.**
- **Designed reusable SQL queries and nearest-neighbor similarity analysis to surface undervalued prospects and communicate data-driven personnel insights.**

## Responsible use

CourtVision is an educational/portfolio project. Model grades are estimates from incomplete public data and should never be represented as official NBA/NCAA evaluations.

## License

MIT
