import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             roc_auc_score, precision_score, roc_curve, auc)

# Direct import of existing models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              AdaBoostClassifier, ExtraTreesClassifier)
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# import XGBoost (optional)
try:
    from xgboost import XGBClassifier
    xgboost_available = True
except ImportError:
    xgboost_available = False

# import LightGBM (optional)
try:
    from lightgbm import LGBMClassifier
    lgbm_available = True
except ImportError:
    lgbm_available = False

# import CatBoost (optional)
try:
    from catboost import CatBoostClassifier
    catboost_available = True
except ImportError:
    catboost_available = False

# =============================
# Setup
# =============================
CONFIG = {
    "data_path": r"C:\path\to\dataset.xlsx",      # Path to the dataset
    "modelo_dir": r"C:\path\to\out",         # Directory to save models and plots
    "random_state": 2718,                                              # Seed for reproducibility
    "n_jobs": -1,                                                      # Number of parallel jobs
    "n_folds": 2,                                                      # Number of splits for GroupKFold
    "target_var": "Progresion",                                        # Target variable
    "normalize": True,                                                 # Whether to normalize numerical variables
    "normalize_method": "zscore",                                      # Normalization method (using StandardScaler)
    "check_multicollinearity": False,
    "multicollinearity_threshold": 5.0,  # VIF value threshold above which multicollinearity is considered problematic
    "feature_selection": False,
    "feature_selection_method": "RFE",  # Or 'SelectKBest'
    "num_features": 30,
    "balance_classes": False,
    "balancing_method": "SMOTE"  # Or 'undersampling'
}

# =============================
# Auxiliary Functions
# =============================

def load_data(data_path):
    """Load the dataset and check that it contains the 'PatientID' column."""
    try:
        data = pd.read_excel(data_path)
    except Exception as e:
        raise IOError(f"Error loading file: {e}")
    if "PatientID" not in data.columns:
        raise ValueError("The 'PatientID' column is not found in the dataset.")
    return data

def create_preprocessor(X, normalize=True):
    """Create the preprocessor by separating numerical and categorical variables."""
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler() if normalize else "passthrough", numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ],
        remainder="passthrough"
    )
    return preprocessor

def get_models(preprocessor):
    """Define a dictionary of pipelines with different models to evaluate."""
    models = {
        "LogisticRegression": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(random_state=CONFIG["random_state"], max_iter=1000))
        ]),
        "SVC": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", SVC(probability=True, random_state=CONFIG["random_state"]))
        ]),
        "RandomForest": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(random_state=CONFIG["random_state"], n_jobs=CONFIG["n_jobs"]))
        ]),
        "GaussianNB": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", GaussianNB())
        ]),
        "DecisionTree": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", DecisionTreeClassifier(random_state=CONFIG["random_state"]))
        ]),
        "KNeighbors": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", KNeighborsClassifier())
        ]),
        "GradientBoosting": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", GradientBoostingClassifier(random_state=CONFIG["random_state"]))
        ]),
        "AdaBoost": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", AdaBoostClassifier(random_state=CONFIG["random_state"], algorithm="SAMME"))
        ]),
        "ExtraTrees": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", ExtraTreesClassifier(random_state=CONFIG["random_state"], n_jobs=CONFIG["n_jobs"]))
        ]),
        "MLP": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", MLPClassifier(random_state=CONFIG["random_state"], max_iter=500))
        ]),
        "LDA": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", LinearDiscriminantAnalysis())
        ])
    }
    if xgboost_available:
        models["XGBoost"] = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", XGBClassifier(random_state=CONFIG["random_state"], eval_metric='logloss'))
        ])
    if lgbm_available:
        models["LightGBM"] = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", LGBMClassifier(random_state=CONFIG["random_state"], n_jobs=CONFIG["n_jobs"]))
        ])
    if catboost_available:
        models["CatBoost"] = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", CatBoostClassifier(random_state=CONFIG["random_state"], verbose=0))
        ])
    return models

def plot_and_save_confusion_matrix(cm, classes, model_name, save_path, title="Confusion Matrix", cmap=plt.cm.Blues):
    """Generate and save the confusion matrix plot."""
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.ylabel("Real label")
    plt.xlabel("Predicted label")
    plt.title(title)
    plt.tight_layout()
    cm_path = os.path.join(save_path, f"confusion_matrix_{model_name}.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Confusion matrix saved at: {cm_path}")

def plot_and_save_roc_curve(y, y_pred_proba, model_name, save_path):
    """Generate and save the global ROC curve plot, and return the AUC."""
    fpr, tpr, _ = roc_curve(y, y_pred_proba[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Global ROC (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Global ROC curve - {model_name}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_path = os.path.join(save_path, f"roc_curve_{model_name}.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"Global ROC curve saved at: {roc_path}")
    return roc_auc

def plot_and_save_metrics(metrics, model_name, save_path):
    """Generate and save a bar plot with the model's metrics."""
    df_metrics = pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])
    plt.figure(figsize=(8, 6))
    ax = sns.barplot(x="Metric", y="Value", data=df_metrics, hue="Metric", dodge=False, palette="viridis")
    plt.ylim(0, 1)
    plt.title(f"Metrics - {model_name}")
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()
    for i, row in df_metrics.iterrows():
        plt.text(i, row["Value"] + 0.02, f"{row['Value']:.2f}", ha="center")
    plt.tight_layout()
    metrics_path = os.path.join(save_path, f"metrics_{model_name}.png")
    plt.savefig(metrics_path, dpi=300)
    plt.close()
    print(f"Metrics plot saved at: {metrics_path}")

def plot_and_save_feature_importances(importances, feature_names, model_name, save_path, top_n=10):
    """Generate and save the feature importances plot.
    Displays only the top_n most important features."""
    # Sort in descending order of importance
    sorted_idx = np.argsort(importances)[::-1]
    sorted_importances = importances[sorted_idx]
    sorted_features = np.array(feature_names)[sorted_idx]
    
    # Take only the top_n features
    top_n = min(top_n, len(sorted_features))
    top_importances = sorted_importances[:top_n]
    top_features = sorted_features[:top_n]
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x=top_importances, y=top_features, palette="viridis")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title(f"Top {top_n} Importance of the characteristics - {model_name}")
    plt.tight_layout()
    fi_path = os.path.join(save_path, f"feature_importances_{model_name}.png")
    plt.savefig(fi_path, dpi=300)
    plt.close()
    print(f"Feature importance plot (top {top_n}) saved at: {fi_path}")


# =============================
# Main Function
# =============================

def main():
    # Create models (and plots) directory if it does not exist
    os.makedirs(CONFIG["modelo_dir"], exist_ok=True)
    
    # Load data
    data = load_data(CONFIG["data_path"])
    print(f"Total data size: {len(data)}")
    print(f"Number of unique patients: {data['PatientID'].nunique()}")
    
    # For validation, 'PatientID' is used as group, but it is removed from the predictors
    target = CONFIG["target_var"]
    X = data.drop(columns=[target, "PatientID"])
    y = data[target]
    
    # Create preprocessor and define models
    preprocessor = create_preprocessor(X, CONFIG["normalize"])
    models = get_models(preprocessor)
    
    # Define the GroupKFold and get the folds
    gkf = GroupKFold(n_splits=CONFIG["n_folds"])
    groups = data["PatientID"]
    folds = list(gkf.split(data, groups=groups))
    fold_assignments = {}
    for i, (train_idx, test_idx) in enumerate(folds):
        fold_assignments[f"Fold {i+1}"] = data.iloc[test_idx]["PatientID"].unique().tolist()
    print("\nPatient assignment in each fold (GroupKFold):")
    for fold, patients in fold_assignments.items():
        print(f"{fold}: {patients}")
    
    # Evaluate each model using cross_validate
    for model_name, pipeline in models.items():
        print(f"\nProcessing model: {model_name}")
        cv_results = cross_validate(
            pipeline, X, y,
            cv=gkf,
            scoring={'accuracy': 'accuracy', 'precision': 'precision', 'roc_auc': 'roc_auc'},
            groups=groups,
            n_jobs=CONFIG["n_jobs"],
            return_estimator=True
        )
        
        # Print average metrics obtained in each fold
        acc_mean = np.mean(cv_results['test_accuracy'])
        prec_mean = np.mean(cv_results['test_precision'])
        auc_mean = np.mean(cv_results['test_roc_auc'])
        acc_std = np.std(cv_results['test_accuracy'])
        prec_std = np.std(cv_results['test_precision'])
        auc_std = np.std(cv_results['test_roc_auc'])
        print(f"Accuracy: {acc_mean:.4f} ± {acc_std:.4f}")
        print(f"Precision: {prec_mean:.4f} ± {prec_std:.4f}")
        print(f"AUC (average per fold): {auc_mean:.4f} ± {auc_std:.4f}")
        
        # Append predictions from each fold using the obtained estimators
        y_pred = np.empty_like(y)
        supports_proba = True
        try:
            cv_results['estimator'][0].predict_proba(X.iloc[folds[0][1]])
        except Exception:
            supports_proba = False
        
        if supports_proba:
            y_pred_proba = np.empty((len(y), 2))
        
        for (train_idx, test_idx), estimator in zip(folds, cv_results['estimator']):
            y_pred[test_idx] = estimator.predict(X.iloc[test_idx])
            if supports_proba:
                y_pred_proba[test_idx] = estimator.predict_proba(X.iloc[test_idx])
        
        # Display classification report based on aggregated predictions
        print(f"\nClassification Report for model {model_name}:")
        print(classification_report(y, y_pred))
        
        # Generate and save the confusion matrix
        cm = confusion_matrix(y, y_pred)
        plot_and_save_confusion_matrix(cm, classes=np.unique(y), model_name=model_name, save_path=CONFIG["modelo_dir"])
        
        # If the model supports predict_proba, plot the global ROC curve and ROC curves for each fold
        if supports_proba:
            # Global ROC
            auc_val = roc_auc_score(y, y_pred_proba[:, 1])
            plot_and_save_roc_curve(y, y_pred_proba, model_name, CONFIG["modelo_dir"])
            
            # Plot the ROC curves for each fold on the same figure and compute the average ROC
            plt.figure(figsize=(8, 6))
            mean_fpr = np.linspace(0, 1, 100)
            tprs = []
            aucs = []
            for i, ((train_idx, test_idx), estimator) in enumerate(zip(folds, cv_results['estimator'])):
                X_test = X.iloc[test_idx]
                y_test = y.iloc[test_idx]
                y_proba_fold = estimator.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_proba_fold)
                fold_auc = auc(fpr, tpr)
                aucs.append(fold_auc)
                # Interpolate tpr to have a common baseline on mean_fpr
                interp_tpr = np.interp(mean_fpr, fpr, tpr)
                interp_tpr[0] = 0.0
                tprs.append(interp_tpr)
                plt.plot(fpr, tpr, lw=2, label=f'Fold {i+1} (AUC = {fold_auc:.2f})')
            # Calculate the average ROC curve
            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = auc(mean_fpr, mean_tpr)
            # Average ROC line (dashed red line)
            plt.plot(mean_fpr, mean_tpr, color='red', linestyle='--', lw=3, label=f'Mean ROC (AUC = {mean_auc:.2f})')
            
            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC curves per fold - {model_name}")
            plt.legend(loc="lower right")
            roc_fold_path = os.path.join(CONFIG["modelo_dir"], f"roc_curves_{model_name}.png")
            plt.savefig(roc_fold_path)
            plt.close()
            print(f"ROC curves per fold saved at: {roc_fold_path}")
        else:
            print(f"Model {model_name} does not support predict_proba, ROC curve is omitted.")
        
        # Plot the average metrics obtained from cross_validate
        metrics_dict = {"accuracy": acc_mean, "precision": prec_mean, "auc": auc_mean}
        plot_and_save_metrics(metrics_dict, model_name, CONFIG["modelo_dir"])
        
        # -------------------------------
        # Extract and plot feature importances
        # -------------------------------
        importances_list = []
        for estimator in cv_results['estimator']:
            clf = estimator.named_steps["classifier"]
            importance = None
            if hasattr(clf, "feature_importances_"):
                importance = clf.feature_importances_
            elif hasattr(clf, "coef_"):
                # For coefficients, take the absolute value
                importance = np.abs(clf.coef_).flatten()
            if importance is not None:
                importances_list.append(importance)
        
        if importances_list:
            avg_importances = np.mean(importances_list, axis=0)
            # Try to obtain the feature names from the preprocessor
            try:
                feature_names = cv_results['estimator'][0].named_steps['preprocessor'].get_feature_names_out()
            except AttributeError:
                # If not available, construct them manually from X
                numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
                categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
                feature_names = np.array(numeric_cols + categorical_cols)
            plot_and_save_feature_importances(avg_importances, feature_names, model_name, CONFIG["modelo_dir"])
        else:
            print(f"Model {model_name} does not provide feature importance.")
            
if __name__ == "__main__":
    main()
