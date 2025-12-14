import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import hydra
from omegaconf import DictConfig
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Допоміжна функція для абсолютних шляхів при використанні Hydra
from hydra.utils import to_absolute_path

def log_confusion_matrix(y_true, y_pred, labels):
    """Будує та логує матрицю плутанини в MLflow"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    filename = "confusion_matrix.png"
    plt.savefig(filename)
    mlflow.log_artifact(filename)
    plt.close()

def log_roc_curve(y_true, y_prob):
    """Будує та логує ROC-криву"""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    
    filename = "roc_curve.png"
    plt.savefig(filename)
    mlflow.log_artifact(filename)
    plt.close()
    
    # Також логуємо AUC як метрику
    mlflow.log_metric("roc_auc", roc_auc)

def log_pr_curve(y_true, y_prob):
    """Будує та логує Precision-Recall криву"""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    
    filename = "pr_curve.png"
    plt.savefig(filename)
    mlflow.log_artifact(filename)
    plt.close()

@hydra.main(version_base=None, config_path="../configs", config_name="main")
def main(cfg: DictConfig):
    # 1. Setup MLflow
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    
    # Використовуємо autolog для захоплення системних метрик, але графіки малюємо вручну
    mlflow.sklearn.autolog(log_models=False, log_datasets=False)

    with mlflow.start_run():
        print(f"Starting run in experiment: {cfg.mlflow.experiment_name}")
        
        # 2. Data Loading
        data_path = to_absolute_path(cfg.paths.raw_data)
        df = pd.read_csv(data_path)
        
        # 3. Data Preprocessing (CRITICAL: Dropping 'odor')
        if cfg.data.drop_columns:
            print(f"Dropping columns: {cfg.data.drop_columns}")
            df = df.drop(columns=list(cfg.data.drop_columns))
        
        X = df.drop(columns=[cfg.data.target_column])
        y = df[cfg.data.target_column]
        
        # Encoding Target (e/p -> 0/1)
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        class_names = le.classes_ # ['e', 'p'] зазвичай
        
        # Split Data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, 
            test_size=cfg.data.test_size, 
            random_state=cfg.project.seed,
            stratify=y_encoded
        )
        
        # 4. Pipeline Construction
        # Для Random Forest достатньо перетворити строки в числа (Ordinal)
        # OneHot може створити забагато колонок, хоча тут це не критично.
        categorical_features = X.columns.tolist()
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_features)
            ],
            remainder='passthrough'
        )
        
        rf = RandomForestClassifier(
            n_estimators=cfg.model.n_estimators,
            max_depth=cfg.model.max_depth,
            min_samples_split=cfg.model.min_samples_split,
            class_weight=cfg.model.class_weight,
            random_state=cfg.project.seed,
            n_jobs=-1
        )
        
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', rf)
        ])
        
        # 5. Training
        print("Training model...")
        pipeline.fit(X_train, y_train)
        
        # 6. Evaluation
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1] # Ймовірність позитивного класу (poisonous)
        
        # Metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='macro'),
            "recall": recall_score(y_test, y_pred, average='macro'),
            "f1_macro": f1_score(y_test, y_pred, average='macro')
        }
        
        print(f"Metrics: {metrics}")
        mlflow.log_metrics(metrics)
        
        # Logging Parameters explicitly
        mlflow.log_params({
            "n_estimators": cfg.model.n_estimators,
            "max_depth": cfg.model.max_depth,
            "dropped_features": str(cfg.data.drop_columns)
        })

        # 7. Artifacts (Visualizations)
        print("Generating artifacts...")
        log_confusion_matrix(y_test, y_pred, class_names)
        log_roc_curve(y_test, y_prob)
        log_pr_curve(y_test, y_prob)
        
        # 8. Save Model
        # Підпис входу/виходу для MLflow Model Registry
        signature = infer_signature(X_train, pipeline.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=pipeline, 
            artifact_path="random_forest_model",
            signature=signature,
            input_example=X_train.iloc[:5]
        )
        
        print("Run complete. Check MLflow UI.")

if __name__ == "__main__":
    main()