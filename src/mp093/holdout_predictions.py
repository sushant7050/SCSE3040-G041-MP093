from pathlib import Path
import joblib
import pandas as pd

from mp093.config import ROOT, load_params
from mp093.data import load_raw

def main():
    params = load_params()
    model = joblib.load(ROOT / "models" / "model_v2.joblib")

    holdout = pd.read_csv(ROOT / "data" / "raw" / "holdout_features.csv")

    record_ids = holdout["record_id"].copy()

    features = params["features"]

    predictions = model.predict(holdout[features])

    output = pd.DataFrame({
        "record_id": record_ids,
        "prediction": predictions
    })

    output_dir = ROOT / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)

    output.to_csv(output_dir / "holdout_predictions.csv", index=False)

    print("Holdout predictions generated successfully.")
    print(f"Rows: {len(output)}")
    print(f"Columns: {list(output.columns)}")
    print(f"Saved to: {output_dir / 'holdout_predictions.csv'}")

if __name__ == "__main__":
    main()