# Telco Customer Churn Prediction Model

Predicts customer churn from usage, spend, and complaint behavior, using
`TRAIN.csv` (labeled, 1,400 customers) and `TEST.csv` (unlabeled, 598
customers). Built with a data-quality gate (correlation + VIF checks),
tuned Logistic Regression and LightGBM, and a SHAP interpretability layer to compare the findings of the model to global benchmarks

## Results

| Model | Holdout ROC-AUC | Holdout Accuracy | 5-fold CV ROC-AUC | 5-fold CV Accuracy |
|---|---|---|---|---|
| Logistic Regression | 0.746 | 0.689 | 0.743 ± 0.028 | 0.681 ± 0.017 |
| **LightGBM (final model)** | **0.900** | **0.807** | **0.901 ± 0.016** | **0.806 ± 0.021** |

the LightGBM model caught ~84% of actual churners on unseen data with ~79% precision, this is good signal (this dataset has genuine behavioral signal: usage, spend, and complaint history)

## Project files

| File | Purpose |
|---|---|
| `churn_model.py` | End-to-end pipeline: clean → correlation/VIF gate → feature engineering → tune → evaluate → save model |
| `shap_explainability.py` | Loads the saved model and produces global SHAP explanations for each customer|
| `churn_model_notebook.ipynb` | The same pipeline in notebook form, cell-by-cell, with inline plots and outputs — already executed, 0 errors |
| `requirements.txt` | Pinned dependencies |
| `TRAIN.csv` / `TEST.csv` | Data Sources |
| `churn_model.joblib` | Saved fitted model + metadata (produced by `churn_model.py`) |
| `train_cleaned.csv` / `test_cleaned.csv` | Cleaned, feature-engineered data (produced by `churn_model.py`, consumed by `shap_explainability.py`) |
| `correlation_heatmap.png` | Correlation matrix of the final, VIF-vetted feature set |
| `shap_summary_plot.png` | SHAP beeswarm plot — direction and magnitude of each driver |
| `test_predictions_with_reasons.csv` | Every `TEST.csv` customer scored, with a plain-language "top 3 reasons" for their risk |

## How to run

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

python3 churn_model.py            # trains, evaluates, saves churn_model.joblib
python3 shap_explainability.py    # must run after churn_model.py , it uses its output
```

Or open `churn_model_notebook.ipynb` in Jupyter to walk through the same pipeline
interactively (`jupyter notebook churn_model_notebook.ipynb`).

## What the data-quality gate does

Before any model is trained, the pipeline automatically:

1. **Correlation check** — drops one feature from any pair with |r| > 0.90.
   This caught `network_age` and `Customer tenure in month`, which turned out to be the
   same information in different units (r = 1.0000) — keeping both would have silently
   doubled their influence and destabilized the Logistic Regression coefficients.
2. **VIF check** — iteratively drops the highest-VIF numeric feature until every
   remaining one is under the standard threshold of 10. Final set: max VIF 6.70
   (total spend), most features under 3.

Both checks run automatically every time the script is re-run on new data, it's a permanent gate in the pipeline.

## Top churn drivers (from SHAP)

1. Total spend (months 1–2)
2. SMS spend
3. Off-network call spend
4. On-network call spend
5. Competitor network preference (month 2)
6. Data consumption, network age/tenure, unique calls, complaint rate

**One pattern worth a business conversation:** very low spend tends to push predicted
risk down while high spend often pushes it up which is the opposite of the usual assumption 
that more engaged users are the most loyal. This could mean high spenders are more price-sensitive
to competitor offers, or that a spend spike happens right before someone leaves (e.g.
burning through a balance). This should be validated. 


## Known limitations

- This is a single labeled snapshot, not a time series so the model predicts if each
  customer resemble past churners and not if a specific customer wil churn in the next
  30 days. A proper time-to-churn model would need monthly snapshots per customer.
