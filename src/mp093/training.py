
"""
MP093 - Stage 2: Model Training and MLflow Tracking

- Loads and cleans train.csv
- Compares three regression models
- Evaluates using MAE
- Tracks experiments in MLflow
- Retrains the best model on all cleaned training data
- Registers the model and assigns the champion alias
"""

from pathlib import Path
import hashlib

import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from mlflow.tracking import MlflowClient

from mp093.cleaning import clean_data


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = ROOT / "data" / "raw" / "train.csv"
REPORT_PATH = ROOT / "reports"

EXPERIMENT_NAME = "MP093-v1"
REGISTERED_MODEL_NAME = "mp093-ration-distribution"

RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "cards_attached",
    "working_hours",
    "biometric_failures",
    "temp_c",
    "stock_kg_morning",
    "days_to_month_end",
]

CATEGORICAL_FEATURES = [
    "shop_area",
    "block",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "grain_kg"


# --------------------------------------------------
# 2. DATA FINGERPRINT
# --------------------------------------------------

def fingerprint(df):

    content = df.to_csv(index=False).encode("utf-8")

    return hashlib.sha256(content).hexdigest()


# --------------------------------------------------
# 3. LOAD AND CLEAN DATA
# --------------------------------------------------

def load_training_data():

    print("\nLoading training dataset...")

    df = pd.read_csv(TRAIN_PATH)

    print("Original shape:", df.shape)

    df = clean_data(df)

    print("Cleaned shape:", df.shape)

    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    print("\nFeature columns:")
    print(X.columns.tolist())

    print("\nTarget:", TARGET)

    print("\nDataset fingerprint:")
    print(fingerprint(df))

    return X, y, df


# --------------------------------------------------
# 4. PREPROCESSING
# --------------------------------------------------

def create_preprocessor():

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )

    return preprocessor


# --------------------------------------------------
# 5. DEFINE THREE MODEL CONFIGURATIONS
# --------------------------------------------------

def get_models():

    return {

        "LinearRegression": LinearRegression(),

        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=2,
            random_state=RANDOM_STATE,
        ),
    }


# --------------------------------------------------
# 6. MODEL COMPARISON WITH MLFLOW
# --------------------------------------------------

def compare_models(X_train, X_test, y_train, y_test):

    print("\nStarting MLflow experiment...")

    mlflow.set_experiment(EXPERIMENT_NAME)

    models = get_models()

    results = []

    trained_models = {}

    for model_name, estimator in models.items():

        print("\n" + "=" * 55)
        print("Training:", model_name)
        print("=" * 55)

        pipeline = Pipeline(
            steps=[
                ("preprocessor", create_preprocessor()),
                ("model", estimator),
            ]
        )

        with mlflow.start_run(run_name=model_name) as run:

            pipeline.fit(X_train, y_train)

            predictions = pipeline.predict(X_test)

            mae = mean_absolute_error(
                y_test,
                predictions,
            )

            print("Validation MAE:", round(mae, 4))

            mlflow.log_param("model_name", model_name)
            mlflow.log_param("random_state", RANDOM_STATE)
            mlflow.log_param("training_rows", len(X_train))
            mlflow.log_param("validation_rows", len(X_test))
            mlflow.log_param("features", ",".join(FEATURES))

            mlflow.log_metric("validation_mae", mae)

            results.append(
                {
                    "model_name": model_name,
                    "validation_mae": mae,
                    "run_id": run.info.run_id,
                }
            )

            trained_models[model_name] = pipeline

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="validation_mae",
        ascending=True,
    )

    REPORT_PATH.mkdir(parents=True, exist_ok=True)

    results_file = REPORT_PATH / "model_comparison.csv"

    results_df.to_csv(results_file, index=False)

    print("\nMODEL COMPARISON")
    print(results_df.to_string(index=False))

    best_model_name = results_df.iloc[0]["model_name"]

    print("\nBest model:", best_model_name)

    return best_model_name, trained_models[best_model_name], results_df


# --------------------------------------------------
# 7. TRAIN BEST MODEL ON FULL CLEANED DATA
# --------------------------------------------------

def train_final_model(best_model_name, X, y, df, results_df):

    print("\nRetraining best model on complete cleaned dataset...")

    models = get_models()

    final_pipeline = Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            ("model", models[best_model_name]),
        ]
    )

    final_pipeline.fit(X, y)

    train_predictions = final_pipeline.predict(X)

    train_mae = mean_absolute_error(
        y,
        train_predictions,
    )

    print("Full training MAE:", round(train_mae, 4))

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="MP093-v1-final") as run:

        mlflow.log_param("selected_model", best_model_name)
        mlflow.log_param("training_rows", len(X))
        mlflow.log_param("feature_count", len(FEATURES))
        mlflow.log_param("random_state", RANDOM_STATE)

        mlflow.log_metric("training_mae", train_mae)

        mlflow.log_param(
            "dataset_fingerprint",
            fingerprint(df),
        )

        mlflow.log_artifact(
            str(REPORT_PATH / "model_comparison.csv"),
            artifact_path="reports",
        )

        # Use cloudpickle to avoid the skops untrusted-type
        # serialization error encountered earlier.
        mlflow.sklearn.log_model(
            sk_model=final_pipeline,
            name="model",
            serialization_format="cloudpickle",
            input_example=X.head(5),
        )

        best_run_id = run.info.run_id

    print("\nFinal MLflow run ID:", best_run_id)

    return final_pipeline, best_run_id, train_mae


# --------------------------------------------------
# 8. REGISTER MODEL AND SET CHAMPION ALIAS
# --------------------------------------------------

def register_champion(best_run_id):

    print("\nRegistering model...")

    model_uri = f"runs:/{best_run_id}/model"

    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=REGISTERED_MODEL_NAME,
    )

    client = MlflowClient()

    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias="champion",
        version=registered_model.version,
    )

    print("\nMODEL REGISTERED SUCCESSFULLY")

    print("Registered model:", REGISTERED_MODEL_NAME)
    print("Version:", registered_model.version)
    print("Alias: champion")

    return registered_model.version


# --------------------------------------------------
# 9. MAIN
# --------------------------------------------------

def main():

    print("\n" + "=" * 60)
    print("MP093 - STAGE 2: MODEL TRAINING")
    print("=" * 60)

    # Load and clean
    X, y, df = load_training_data()

    # Validation split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    print("\nTraining rows:", len(X_train))
    print("Validation rows:", len(X_test))

    # Compare three models
    best_model_name, _, results_df = compare_models(
        X_train,
        X_test,
        y_train,
        y_test,
    )

    # Train final v1 on all cleaned training data
    final_model, best_run_id, train_mae = train_final_model(
        best_model_name,
        X,
        y,
        df,
        results_df,
    )

    # Register champion
    champion_version = register_champion(best_run_id)

    # Final summary
    print("\n" + "=" * 60)
    print("STAGE 2 TRAINING COMPLETE")
    print("=" * 60)

    print("Experiment:", EXPERIMENT_NAME)
    print("Best model:", best_model_name)
    print("Training MAE:", round(train_mae, 4))
    print("Best run ID:", best_run_id)
    print("Registered model:", REGISTERED_MODEL_NAME)
    print("Champion version:", champion_version)

    print("\nModel comparison:")
    print(results_df.to_string(index=False))

    print("\nNext: Verify the MLflow run and registered model.")


if __name__ == "__main__":
    main()
