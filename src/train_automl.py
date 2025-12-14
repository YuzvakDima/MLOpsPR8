import pandas as pd
import numpy as np
import time
import hydra
import pickle  # Необхідно для ручного збереження моделі
from omegaconf import DictConfig
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from flaml import AutoML
from hydra.utils import to_absolute_path

@hydra.main(version_base=None, config_path="../configs", config_name="main")
def main(cfg: DictConfig):
    # 1. Setup MLflow
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    # Додаємо суфікс _automl до імені експерименту
    experiment_name = f"{cfg.mlflow.experiment_name}_automl"
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run():
        print(f"Starting AutoML run in experiment: {experiment_name}")
        
        # 2. Data Loading
        data_path = to_absolute_path(cfg.paths.raw_data)
        df = pd.read_csv(data_path)
        
        # 3. Preprocessing (Consistency with Baseline)
        if cfg.data.drop_columns:
            print(f"Dropping columns: {cfg.data.drop_columns}")
            df = df.drop(columns=list(cfg.data.drop_columns))
        
        X = df.drop(columns=[cfg.data.target_column])
        y = df[cfg.data.target_column]
        
        # Encoding Target
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        # Split Data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, 
            test_size=cfg.data.test_size, 
            random_state=cfg.project.seed,
            stratify=y_encoded
        )
        
        # 4. Configure AutoML
        automl = AutoML()
        automl_settings = {
            "time_budget": cfg.automl.time_budget,
            "metric": cfg.automl.metric,
            "task": cfg.automl.task,
            "seed": cfg.project.seed,
            "log_file_name": "flaml.log",
            "verbose": 0
        }
        
        print(f"Starting search with time budget: {cfg.automl.time_budget}s ...")
        start_time = time.time()
        
        # 5. Train (Search)
        automl.fit(X_train=X_train, y_train=y_train, **automl_settings)
        
        duration = time.time() - start_time
        
        # 6. Evaluation
        print("Search completed. Evaluating best model...")
        y_pred = automl.predict(X_test)
        y_prob = automl.predict_proba(X_test)[:, 1]
        
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_macro": f1_score(y_test, y_pred, average='macro'),
            "roc_auc": roc_auc_score(y_test, y_prob),
            "training_duration": duration
        }
        
        print(f"Metrics: {metrics}")
        mlflow.log_metrics(metrics)
        
        # 7. Logging Best Model Info
        best_algo = automl.best_estimator
        best_config = automl.best_config
        
        print(f"Winner Algorithm: {best_algo}")
        print(f"Best Hyperparameters: {best_config}")
        
        mlflow.log_param("best_estimator", best_algo)
        for k, v in best_config.items():
            mlflow.log_param(f"best_{k}", v)

        # 8. Save Model (FINAL FIX)
        # Ми зберігаємо automl.model (саму модель), а не automl (весь інструмент пошуку).
        # Це відкидає логери і потоки, які викликають помилку.
        
        filename = "best_automl_model.pkl"
        
        with open(filename, "wb") as f:
            # Зберігаємо ТІЛЬКИ знайдену модель (наприклад, XGBoost), а не весь FLAML
            pickle.dump(automl.model, f)
            
        mlflow.log_artifact(filename)
        
        print(f"Run complete. Winner: {best_algo}. Model saved manually to MLflow artifacts.")

if __name__ == "__main__":
    main()