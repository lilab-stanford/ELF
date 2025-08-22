import os
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import argparse


def load_aneuploidy_data(feature_dirs):
    """Load TCGA AneuploidyScore dataset, supports multiple model ensemble."""
    # Load data splits
    base_path = os.getcwd()
    train_df = pd.read_csv(os.path.join(base_path, "TCGA-AneuploidyScore_train.csv"))
    val_df = pd.read_csv(os.path.join(base_path, "TCGA-AneuploidyScore_val.csv"))
    test_df = pd.read_csv(os.path.join(base_path, "TCGA-AneuploidyScore_test.csv"))

    # Set feature_dirs as list for easy handling of single model and ensemble
    if isinstance(feature_dirs, str):
        feature_dirs = [feature_dirs]

    # Get all slide IDs
    all_slide_ids = set()
    for df in [train_df, val_df, test_df]:
        all_slide_ids.update(df['Slide'].tolist())

    # Load features
    features_dict = {}
    for slide_id in tqdm(all_slide_ids, desc="Loading features"):
        features_list = []
        for feature_dir in feature_dirs:
            # Extract filename from complete slide ID
            # Example: TCGA-G3-A3CJ-01Z-00-DX1.80279A8A-38D9-4B23-8A83-E32D8C4FC6EA
            # Need to find corresponding h5 file
            h5_path = os.path.join(feature_dir, f"{slide_id}.h5")

            if os.path.exists(h5_path):
                try:
                    with h5py.File(h5_path, 'r') as f:
                        # For slide_embedding models, use features key (768 dimensions)
                        # For other models, select according to situation
                        if ("titan" in feature_dir or
                                "titan_elf" in feature_dir or
                                "prov_gigapath" in feature_dir or "chief" in feature_dir):
                            feat = f["features"][:][0]
                        else:
                            feat = f["features_dim"][:][0]

                        # Standardize individual features
                        feat = (feat - np.mean(feat)) / np.std(feat)
                        features_list.append(feat)
                except Exception as e:
                    print(f"Error loading {h5_path}: {str(e)}")

        if features_list:
            # Process features
            processed_features = np.concatenate(features_list, axis=0)
            features_dict[slide_id] = processed_features

    print(f"Successfully loaded features for {len(features_dict)} samples")

    # Add features and labels to dataframes
    for df in [train_df, val_df, test_df]:
        df['features'] = df['Slide'].map(features_dict)
        df['label'] = df['Genome_doublings']  # Directly use AneuploidyScore as label
        df['label'] = (df['label'].astype(int) > 0).astype(int) # WGD 0 vs WGD 1 and 2

        # Remove rows without features
        df.dropna(subset=['features'], inplace=True)

    # Print dataset size and class distribution
    print(f"\nDataset statistics:")
    print(f"Training set samples: {len(train_df)}")
    print(f"Validation set samples: {len(val_df)}")
    print(f"Test set samples: {len(test_df)}")
    print(f"Total samples: {len(train_df) + len(val_df) + len(test_df)}")

    if len(features_dict) > 0:
        feature_shape = next(iter(features_dict.values())).shape
        print(f"Feature shape: {feature_shape}")

    return train_df, val_df, test_df


def evaluate_aneuploidy(train_data, val_data, test_data, max_iter=2000, model_name="chief"):
    """Evaluate AneuploidyScore classification model."""

    def check_features(data, name=""):
        """Check and report feature shapes, output problematic slide IDs."""
        shapes = []
        problem_slides = []

        # Find most common shape
        all_shapes = [feat.shape for feat in data['features'] if isinstance(feat, np.ndarray)]
        if not all_shapes:
            return shapes, problem_slides

        most_common_shape = max(set(all_shapes), key=all_shapes.count)

        # Check each feature
        for idx, (slide_id, feat) in enumerate(zip(data['Slide'], data['features'])):
            if not isinstance(feat, np.ndarray):
                problem_slides.append((slide_id, f"Not numpy array, but {type(feat)}"))
                continue

            if feat.shape != most_common_shape:
                problem_slides.append((slide_id, f"Shape is {feat.shape}, doesn't match common shape {most_common_shape}"))
            shapes.append(feat.shape)

        # Report different shapes
        unique_shapes = set(str(s) for s in shapes)

        return shapes, problem_slides

    # Check features in each dataset
    train_shapes, train_problems = check_features(train_data, "Training set")
    val_shapes, val_problems = check_features(val_data, "Validation set")
    test_shapes, test_problems = check_features(test_data, "Test set")

    # If there are problematic slides, summarize report
    all_problems = train_problems + val_problems + test_problems
    if all_problems:
        print("\nTotal problematic slides found:", len(all_problems))
        print("Please fix these issues before continuing")
        return None

    # Try to stack features
    try:
        X_train = np.stack(train_data['features'].values)
        X_val = np.stack(val_data['features'].values)
        X_test = np.stack(test_data['features'].values)

    except ValueError as e:
        print("\nError: Cannot stack features, reason:", str(e))
        print("Please check the above feature shape statistics to ensure all feature dimensions are consistent")
        raise

    y_train = train_data['label'].values
    y_val = val_data['label'].values
    y_test = test_data['label'].values

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Find best model
    best_score = -float('inf')
    best_model = None
    C_values = np.logspace(-10, 10, num=21)

    print("\nFinding best hyperparameters...")
    for C in tqdm(C_values, desc="Finding best C value"):
        model = LogisticRegression(
            C=C,
            fit_intercept=True,
            class_weight='balanced',
            max_iter=max_iter,
            random_state=42,
            solver="lbfgs",
            tol=1e-4
        )
        model.fit(X_train, y_train)
        val_preds = model.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, val_preds)

        if score > best_score:
            best_score = score
            best_model = model

    print(f"Best validation set AUC: {best_score:.4f}")

    # Evaluate final model on test set
    test_preds = best_model.predict_proba(X_test)[:, 1]
    val_preds = best_model.predict_proba(X_val)[:, 1]
    metrics = {
        'auc': roc_auc_score(y_test, test_preds),
    }

    # Validation set results (slide level)
    val_results = pd.DataFrame({
        'Sample': val_data['Sample'].values,
        'Slide': val_data['Slide'].values,
        'True_Label': y_val,
        'Prediction': val_preds,
        'Split': 'validation'
    })

    # Test set results (slide level)
    test_results = pd.DataFrame({
        'Sample': test_data['Sample'].values,
        'Slide': test_data['Slide'].values,
        'True_Label': y_test,
        'Prediction': test_preds,
        'Split': 'test'
    })

    # Patient-level aggregation and evaluation
    print(f"\n=== Patient-level Aggregated Evaluation ===")

    # Validation set patient-level aggregation
    val_patient_agg = val_results.groupby('Sample').agg({
        'True_Label': 'first',  # Each patient's label should be consistent
        'Prediction': 'mean',  # Take mean of predictions
    }).reset_index()

    # Test set patient-level aggregation
    test_patient_agg = test_results.groupby('Sample').agg({
        'True_Label': 'first',
        'Prediction': 'mean'
    }).reset_index()

    print(f"Validation set: {len(val_results)} slides -> {len(val_patient_agg)} patients")
    print(f"Test set: {len(test_results)} slides -> {len(test_patient_agg)} patients")

    base_path = os.path.join(os.getcwd(), "wgd_cls")
    os.makedirs(base_path, exist_ok=True)

    # Save slide-level results
    all_results_slide = pd.concat([val_results, test_results], ignore_index=True)
    output_file_slide = f"wgd_predictions_{model_name}_lg_auc_optimized_slide_level.csv"
    all_results_slide.to_csv(os.path.join(base_path, output_file_slide), index=False)

    # Save patient-level aggregated results
    val_patient_agg['Split'] = 'validation'
    test_patient_agg['Split'] = 'test'
    all_results_patient = pd.concat([val_patient_agg, test_patient_agg],
                                   ignore_index=True)
    
    output_file_patient = os.path.join(base_path, f"wgd_predictions_{model_name}_lg_auc_optimized_patient_level.csv")
    all_results_patient.to_csv(output_file_patient, index=False)

    # Print results
    print("\n===== Test Set Results =====")
    val_patient_auc = roc_auc_score(val_patient_agg['True_Label'],
                                    val_patient_agg['Prediction'])
    test_patient_auc = roc_auc_score(test_patient_agg['True_Label'],
                                     test_patient_agg['Prediction'])
    print(f"Validation set AUC: {val_patient_auc:.4f}")
    print(f"Test set AUC: {test_patient_auc:.4f}")

    return metrics


def run_single_model_evaluation(model_name, max_iter=2000):
    """Run single model evaluation."""
    base_path = os.getcwd()
    feature_dir = os.path.join(base_path, model_name)

    print(f"\n===== Evaluating single model: {model_name} =====")
    print(f"Loading features from: {feature_dir}")

    # Load data
    train_data, val_data, test_data = load_aneuploidy_data(feature_dir)

    # Evaluate model
    results = evaluate_aneuploidy(train_data, val_data, test_data, max_iter, model_name)

    return results


def run_ensemble_evaluation(model_names, max_iter=2000):
    """Run ensemble model evaluation."""
    base_path = os.getcwd()
    # Build feature directory list
    feature_dirs = []
    for model_name in model_names:
        feature_dir = os.path.join(base_path, model_name)
        feature_dirs.append(feature_dir)

    ensemble_name = "_".join([m.split("_")[0] for m in model_names])
    print(f"\n===== Evaluating ensemble model: {ensemble_name} =====")
    print(f"Models: {model_names}")

    # Load ensemble features
    train_data, val_data, test_data = load_aneuploidy_data(feature_dirs)

    # Evaluate ensemble model
    results = evaluate_aneuploidy(train_data, val_data, test_data, max_iter,
                                 model_name="omnipath_elf")

    return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='TCGA AneuploidyScore classification task (supports single model and ensemble)'
    )
    parser.add_argument('--max_iter', type=int, default=10000, help='Logistic regression maximum iterations')

    args = parser.parse_args()

    # Single model evaluation
    for model_name in ["titan", "chief", "prov_gigapath"]:
        results = run_single_model_evaluation(
            model_name,
            max_iter=args.max_iter
        )

    # Ensemble model evaluation
    results = run_ensemble_evaluation(
        model_names=["uni_elf", "virchow2_elf", "conch_v1_5_elf", "h0_elf", "gigapath_elf"],
        max_iter=args.max_iter)

    print("\n===== Evaluation Complete =====")
    return results


if __name__ == "__main__":
    main() 