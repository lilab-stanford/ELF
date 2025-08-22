import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import h5py
import random
from tqdm import tqdm
from collections import Counter
from scipy.interpolate import interp1d
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, roc_auc_score
)
import argparse

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")


def load_aneuploidy_data(feature_dirs):
    """Load aneuploidy data from feature directories."""
    base = os.getcwd()
    dfs = {
        split: pd.read_csv(os.path.join(base, f"TCGA-AneuploidyScore_{split}.csv"))
        for split in ("train", "val", "test")
    }

    if isinstance(feature_dirs, str):
        feature_dirs = [feature_dirs]

    # Map slide_id -> features
    slide_ids = set().union(*(df['Slide'] for df in dfs.values()))
    feats = {}
    for sid in tqdm(slide_ids, desc="Loading features"):
        flist = []
        for d in feature_dirs:
            path = os.path.join(d, f"{sid}.h5")
            if not os.path.exists(path):
                continue
            with h5py.File(path, 'r') as f:
                if ("titan" in path or "chief" in path or 
                        'prov_gigapath' in path):
                    arr = f["features"][:] 
                else:
                    arr = f["features_dim"][:][0]
                if arr.ndim == 2 and arr.shape[0] == 1:
                    arr = arr[0]
                flist.append((arr - arr.mean()) / (arr.std() + 1e-6))
        if flist:
            feats[sid] = np.concatenate(flist, axis=0)

    for split, df in dfs.items():
        df['features'] = df['Slide'].map(feats)
        df['label'] = df['AneuploidyScore']
        dfs[split] = df.dropna(subset=['features']).reset_index(drop=True)

    print("\nDataset sizes:")
    for split, df in dfs.items():
        print(f"{split}: {len(df)} samples, labels {df['label'].min()}–{df['label'].max()}")

    return dfs['train'], dfs['val'], dfs['test']


def evaluate_regression(train_df, val_df, test_df, model_names, model_type="rf"):
    """Evaluate regression model performance."""
    # Stack features
    X_train = np.stack(train_df['features'].values)
    X_val = np.stack(val_df['features'].values)
    X_test = np.stack(test_df['features'].values)

    y_train = train_df['label'].values 
    y_val = val_df['label'].values
    y_test = test_df['label'].values
    
    # Calculate median of all labels for binary classification
    all_labels = np.concatenate([y_train, y_val, y_test])
    # label_median = np.median(all_labels)
    label_median = 16
    print(f"Label median value: {label_median:.2f}")
    
    # Convert labels to binary classification (0/1)
    y_train_binary = (y_train >= label_median).astype(int)
    y_val_binary = (y_val >= label_median).astype(int)
    y_test_binary = (y_test >= label_median).astype(int)
    
    print(f"Binary label distribution:")
    print(f"  Train: {np.sum(y_train_binary)}/{len(y_train_binary)} "
          f"({np.mean(y_train_binary):.3f})")
    print(f"  Val:   {np.sum(y_val_binary)}/{len(y_val_binary)} "
          f"({np.mean(y_val_binary):.3f})")
    print(f"  Test:  {np.sum(y_test_binary)}/{len(y_test_binary)} "
          f"({np.mean(y_test_binary):.3f})")
    
    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Select parameter grid and model class based on model type
    if model_type.lower() in ["xgb", "xgboost"]:
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not available. Install with: pip install xgboost")
        
        print(f"\nStarting grid search optimization for XGBoostRegressor parameters...")
        model_class = xgb.XGBRegressor
        param_grid = {
            'n_estimators': [1000],
            'max_depth': [10],
            'learning_rate': [0.03, 0.05],
            'subsample': [0.8],
            'colsample_bytree': [0.8]
        }
        fixed_params = {'random_state': 42, 'n_jobs': -1}
    else:  # Default to RandomForest
        print(f"\nStarting grid search optimization for RandomForestRegressor parameters...")
        model_class = RandomForestRegressor
        param_grid = {
            'n_estimators': [100, 200, 500, 1000],
            'max_depth': [None, 10, 20],
            'max_features': ['sqrt', 0.2, 0.5],
            'min_samples_leaf': [1, 5, 10]
        }
        fixed_params = {'random_state': 42, 'n_jobs': -1}
    
    best_cfg = None
    best_auc = -1
    best_mse = float('inf')
    best_r = -1

    # Prediction value normalizer (for AUC calculation)
    pred_scaler = MinMaxScaler()

    # Loop search on validation set
    print(f"Total of {len(list(ParameterGrid(param_grid)))} parameter combinations to try...")
    best_model = None
    for i, cfg in enumerate(ParameterGrid(param_grid)):
        # Merge fixed parameters and search parameters
        model_params = {**fixed_params, **cfg}
        model = model_class(**model_params)
        model.fit(X_train, y_train)
        
        preds_val = model.predict(X_val)
        
        # Regression metrics
        mse = mean_squared_error(y_val, preds_val)
        r, _ = pearsonr(y_val, preds_val)
        
        # AUC calculation: normalize predictions to 0-1
        preds_val_normalized = pred_scaler.fit_transform(
            preds_val.reshape(-1, 1)
        ).flatten()
        auc = roc_auc_score(y_val_binary, preds_val_normalized)
        
        print(f"[{i+1:2d}] {cfg}: val AUC={auc:.4f}, MSE={mse:.4f}, "
              f"Pearson r={r:.4f}")
        
        # Select optimal parameters based on AUC as primary metric
        if auc > best_auc:
            best_auc = auc
            best_mse = mse
            best_r = r
            best_cfg = cfg
            best_model = model

    print(f"\nOptimal parameters on validation set: {best_cfg}")
    print(f"Corresponding val AUC={best_auc:.4f}, MSE={best_mse:.4f}, "
          f"Pearson r={best_r:.4f}")

    # Final evaluation on test set
    preds_test = best_model.predict(X_test)
    preds_val_final = best_model.predict(X_val)
    
    # Save slide-level prediction results
    model_name_str = "_".join(model_names)
    
    # AUC metrics: normalize predictions to 0-1
    preds_test_normalized = pred_scaler.fit_transform(
        preds_test.reshape(-1, 1)
    ).flatten()
    preds_val_final_normalized = pred_scaler.fit_transform(
        preds_val_final.reshape(-1, 1)
    ).flatten()
    
    # Validation set results (slide level)
    val_results = pd.DataFrame({
        'Sample': val_df['Sample'].values,
        'Slide': val_df['Slide'].values,
        'True_Label': y_val,
        'True_Label_Binary': y_val_binary,
        'Prediction': preds_val_final,
        'Prediction_Normalized': preds_val_final_normalized,
        'Split': 'validation'
    })
    
    # Test set results (slide level)
    test_results = pd.DataFrame({
        'Sample': test_df['Sample'].values,
        'Slide': test_df['Slide'].values,
        'True_Label': y_test,
        'True_Label_Binary': y_test_binary,
        'Prediction': preds_test,
        'Prediction_Normalized': preds_test_normalized,
        'Split': 'test'
    })
    
    # Patient-level aggregation and evaluation
    print(f"\n=== Patient-level Aggregated Evaluation ===")
    
    # Validation set patient-level aggregation
    val_patient_agg = val_results.groupby('Sample').agg({
        'True_Label': 'first',  # Each patient's label should be consistent
        'True_Label_Binary': 'first',
        'Prediction': 'mean',  # Take mean of predictions
        'Prediction_Normalized': 'mean'  # Take mean of normalized predictions
    }).reset_index()
    
    # Test set patient-level aggregation
    test_patient_agg = test_results.groupby('Sample').agg({
        'True_Label': 'first',
        'True_Label_Binary': 'first', 
        'Prediction': 'mean',
        'Prediction_Normalized': 'mean'
    }).reset_index()
    
    print(f"Validation set: {len(val_results)} slides -> {len(val_patient_agg)} patients")
    print(f"Test set: {len(test_results)} slides -> {len(test_patient_agg)} patients")
    
    # Calculate patient-level performance metrics
    
    # Validation set patient-level metrics
    val_patient_auc = roc_auc_score(
        val_patient_agg['True_Label_Binary'], 
        val_patient_agg['Prediction_Normalized']
    )
    val_patient_mse = mean_squared_error(
        val_patient_agg['True_Label'], 
        val_patient_agg['Prediction']
    )
    val_patient_r, _ = pearsonr(
        val_patient_agg['True_Label'], 
        val_patient_agg['Prediction']
    )
    
    # Test set patient-level metrics  
    test_patient_auc = roc_auc_score(
        test_patient_agg['True_Label_Binary'], 
        test_patient_agg['Prediction_Normalized']
    )
    test_patient_mse = mean_squared_error(
        test_patient_agg['True_Label'], 
        test_patient_agg['Prediction']
    )
    test_patient_r, _ = pearsonr(
        test_patient_agg['True_Label'], 
        test_patient_agg['Prediction']
    )
    
    print(f"\n=== Validation Set Results (Patient Level) ===")
    print(f"  AUC:      {val_patient_auc:.4f}")
    print(f"  MSE:      {val_patient_mse:.4f}")
    print(f"  Pearson:  {val_patient_r:.4f}")
    
    print(f"\n=== Test Set Results (Patient Level) ===")
    print(f"  AUC:      {test_patient_auc:.4f}")
    print(f"  MSE:      {test_patient_mse:.4f}")
    print(f"  RMSE:     {np.sqrt(test_patient_mse):.4f}")
    print(f"  MAE:      {mean_absolute_error(test_patient_agg['True_Label'], test_patient_agg['Prediction']):.4f}")
    print(f"  R2:       {r2_score(test_patient_agg['True_Label'], test_patient_agg['Prediction']):.4f}")
    print(f"  Pearson:  {test_patient_r:.4f}")
    
    # Generate filename including model type
    model_type_suffix = "xgb" if model_type.lower() in ["xgb", "xgboost"] else "rf"
    
    # Save slide-level results
    all_results_slide = pd.concat([val_results, test_results], ignore_index=True)
    output_file_slide = (f"aneuploidy_predictions_{model_name_str}_{model_type_suffix}"
                        f"_auc_optimized_slide_level.csv")
    all_results_slide.to_csv(output_file_slide, index=False)
    
    # Save patient-level aggregated results
    val_patient_agg['Split'] = 'validation'
    test_patient_agg['Split'] = 'test'
    all_results_patient = pd.concat([val_patient_agg, test_patient_agg], 
                                   ignore_index=True)
    output_file_patient = (f"aneuploidy_predictions_{model_name_str}_{model_type_suffix}"
                          f"_auc_optimized_patient_level.csv")
    base_path = os.getcwd()
    all_results_patient.to_csv(os.path.join(base_path, output_file_patient), index=False)
    
    print(f"\nPrediction results saved:")
    print(f"  Slide level: {output_file_slide}")
    print(f"  Patient level: {output_file_patient}")
    print(f"Validation set samples: {len(val_results)} slides / {len(val_patient_agg)} patients")
    print(f"Test set samples: {len(test_results)} slides / {len(test_patient_agg)} patients")
    print(f"Fields included: original labels, binary labels, original predictions, normalized predictions")

    return best_model


def run_single_model_evaluation(model_names, model_type="rf"):
    """Run single model evaluation."""
    base = os.getcwd()
    
    # Handle both single model name (string) and multiple model names (list)
    if isinstance(model_names, str):
        model_names = [model_names]
    
    # Build feature directories for all models
    feature_dirs = [os.path.join(base, model_name) for model_name in model_names]
    
    model_type_name = ("XGBoostRegressor" if model_type.lower() in ["xgb", "xgboost"] 
                       else "RandomForestRegressor")
    print(f"\nEvaluating {model_type_name} on models: {', '.join(model_names)}")
    print(f"Feature directories: {feature_dirs}")
    
    train_df, val_df, test_df = load_aneuploidy_data(feature_dirs)
    model = evaluate_regression(
        train_df, val_df, test_df,
        model_names,
        model_type=model_type
    )
    return model

def main():
    """Main function to run evaluation."""
    parser = argparse.ArgumentParser(
        description="TCGA AneuploidyScore regression task (supports RandomForest and XGBoost)"
    )
    parser.add_argument(
        "--model", type=str, default="titan",
        help="Feature directory name, supports single model or model list for fusion"
    )
    
    parser.add_argument(
        "--model_type", type=str, default="xgb",
        choices=["rf", "randomforest", "xgb", "xgboost"],
        help="Model type: rf/randomforest or xgb/xgboost"
    )
    # Keep these parameters for compatibility, but now automatically optimized via grid search
    parser.add_argument(
        "--max_depth", type=int, default=10,
        help="Model max_depth (now automatically optimized via grid search)"
    )
    parser.add_argument(
        "--n_estimators", type=int, default=1000,
        help="Model n_estimators (now automatically optimized via grid search)"
    )   
    args = parser.parse_args()
    model_list = ["titan", "chief", "prov_gigapath"]
    
    for model_name in model_list:
        run_single_model_evaluation(
            model_name,
            args.model_type
        )
    elf_list = ["virchow2_elf", "conch_v1_5_elf", "h0_elf", "gigapath_elf", "uni_elf"]
    run_single_model_evaluation(
        elf_list,
        args.model_type
    )
     
    
    print("\nDone.")


if __name__ == "__main__":
    main()