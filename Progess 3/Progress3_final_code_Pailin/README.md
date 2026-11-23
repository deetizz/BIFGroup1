# BIF633/CSC345 Group 2 – Progress 3 (2nd round models)

Predict chlorophyll-a at Scripps Pier one week ahead:
- **value** of Chl-a next week (regression)
- **bloom or not** next week (classification; bloom = Chl-a above 6.2 mg/m³, the top 10% of training weeks)

Every figure in `BIF633_Progress3_PP.pptx` that shows our results is made by this code.

## How to run

1. Put `HABs_ScrippsPier_v3.xlsx` next to the `code/` folder.
2. Install the packages: `pip install pandas numpy scikit-learn scipy matplotlib openpyxl`.
3. Run the scripts **in order** from inside `code/`:

| # | Script | What it does | Slide | Time |
|---|---|---|---|---|
| – | `common.py` | Reads the data, fills missing values (past weeks only), builds the inputs, splits by time 70/15/15 | – | – |
| – | `style.py` | Colours and fonts shared by all figures | – | – |
| 1 | `01_progress1_summary.py` | Progress 1 in one figure: skewed, spring peak, 5-week memory | 3 | < 1 min |
| 2 | `02_regression_v2.py` | Regression: 5 input sets × 3 models, choose on validation, test once | 5, 8 | ~5 min |
| 3 | `03_bloom_classification.py` | Bloom or not: 4 models × 2 input sets, choose by cost, test once | 6, 8 | ~6 min |
| 4 | `04_mlp_training_curves.py` | Neural network training curves (small vs large) | 7 | ~1 min |
| 5 | `05_clustering.py` | k-means water types, silhouette, bloom rate per type | 7 | < 1 min |
| 6 | `06_slide_figures.py` | Figures for slides 5, 6 and 8 from the result tables | 5, 6, 8 | < 1 min |

Script 3 needs a table from script 2, and script 6 needs tables from 2 and 3.

## Outputs

`results/` – result tables (CSV) and run logs.
`results/slide_figures/` – the figures used in the slides:

| File | Slide |
|---|---|
| `s03_progress1_summary.png` | 3 · From Progress 1 to Final |
| `s05_regression_validation.png` | 5 · Regression, round 2 |
| `s06_bloom_cost.png` | 6 · Bloom classification |
| `s07a_mlp_training_curves.png` | 7 · Why are blooms hard to predict? (left) |
| `s07b_clusters_bloom_rate.png` | 7 · Why are blooms hard to predict? (right) |
| `s08_final_test.png` | 8 · Final test, used once |

Not from this code: the photos on slide 2 (sources on the slide), the Progress 2 figure on slide 3 (Progress 2 code), and the workflow diagram on slide 4.

## Methods and where they come from

| Step | Method | Course |
|---|---|---|
| Split | by time: train 361 / validation 77 / test 78 weeks; test used once | Lecture 9 (holdout) |
| Tuning | grid search with time-series cross-validation (5 folds), train only | Lecture 9, Lecture 15 |
| Regression | Linear Regression, Random Forest, SVR; error = RMSE (mg/m³) | Lectures 12–13, 15 |
| Bloom | Logistic Regression, Random Forest, SVM, MLP (class weights for the rare class) | Lectures 14, 15 |
| Bloom scores | confusion matrix, precision, recall, F-measure, ROC-AUC, cost matrix (missed bloom = 5, false alarm = 1) | Lecture 9 |
| Explain | training curves; k-means + silhouette, hierarchical (Ward) | Lecture 14; Lectures 12–13 |
| Baselines | persistence (next week = this week), train median, seasonal median, never bloom | – |

## Changes from the first Progress 3 code (`bif633_pipeline_v2.py`)

1. Missing values: no interpolation over the whole series (it used later weeks). Now: last known value, then the train median.
2. Input sets picked from validation results (D_slide3, E_slide3_chl) removed.
3. Models are compared on validation only; the test set is used once, for the chosen model.
4. The bloom script is included (it was missing) and reproduces the earlier bloom numbers.
5. New: SVM with class weights, MLP, training curves, clustering.
