import os
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import joblib


ebrains_label_dict = {
    'Glioblastoma, IDH-wildtype': 0,
    'Transitional meningioma': 1,
    'Anaplastic meningioma': 2,
    'Pituitary adenoma': 3,
    'Oligodendroglioma, IDH-mutant and 1p/19q codeleted': 4,
    'Haemangioma': 5,
    'Ganglioglioma': 6,
    'Schwannoma': 7,
    'Anaplastic oligodendroglioma, IDH-mutant and 1p/19q codeleted': 8,
    'Anaplastic astrocytoma, IDH-wildtype': 9,
    'Pilocytic astrocytoma': 10,
    'Angiomatous meningioma': 11,
    'Haemangioblastoma': 12,
    'Gliosarcoma': 13,
    'Adamantinomatous craniopharyngioma': 14,
    'Anaplastic astrocytoma, IDH-mutant': 15,
    'Ependymoma': 16,
    'Anaplastic ependymoma': 17,
    'Glioblastoma, IDH-mutant': 18,
    'Atypical meningioma': 19,
    'Metastatic tumours': 20,
    'Meningothelial meningioma': 21,
    'Langerhans cell histiocytosis': 22,
    'Diffuse large B-cell lymphoma of the CNS': 23,
    'Diffuse astrocytoma, IDH-mutant': 24,
    'Secretory meningioma': 25,
    'Haemangiopericytoma': 26,
    'Fibrous meningioma': 27,
    'Lipoma': 28,
    'Medulloblastoma, non-WNT/non-SHH': 29
}


def load_ebrains_data(feature_dir):
    """Load ebrains dataset with features and labels"""
    # Load splits
    base_path = os.getcwd()
    train_df = pd.read_csv(os.path.join(base_path, "train.csv"))
    val_df = pd.read_csv(os.path.join(base_path, "val.csv"))
    test_df = pd.read_csv(os.path.join(base_path, "test.csv"))

    # Load features
    features_dict = {}
    for file_name in os.listdir(feature_dir):
        if file_name.endswith('.h5'):
            slide_id = file_name.replace('.h5', '')
            with h5py.File(os.path.join(feature_dir, file_name), 'r') as f:
                if any(model in feature_dir for model in [
                    "titan", "prov_gigapath", "chief", "gigapath_mean_pooling",
                    "virchow2_mean_pooling", "h0_mean_pooling", "uni_mean_pooling",
                    "conch_v1_5_mean_pooling", "gigapath_max_pooling",
                    "virchow2_max_pooling", "h0_max_pooling", "uni_max_pooling",
                    "conch_v1_5_max_pooling"
                ]):
                    feat = f["features"][:][0]
                    feat = (feat - np.mean(feat)) / np.std(feat)
                    features_dict[slide_id] = feat
                else:
                    feat = f["features_dim"][:][0]
                    feat = (feat - np.mean(feat)) / np.std(feat)
                    features_dict[slide_id] = feat

    # Add features and labels
    for df in [train_df, val_df, test_df]:
        df['features'] = df['slide_id'].map(features_dict)
        df['label'] = df['diagnosis'].map(ebrains_label_dict)
        df = df.dropna(subset=['features'])

    # Print dataset size
    print(f"Training set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")
    print(f"Test set size: {len(test_df)}")
    print(f"Total set size: {len(train_df) + len(val_df) + len(test_df)}")

    return train_df, val_df, test_df


def bootstrap(targets_all, preds_all, probs_all=None, n=1000, alpha=0.95):
    """
    Calculate bootstrap statistics for evaluation metrics.
    
    Args:
        targets_all: Ground truth labels
        preds_all: Predicted labels
        probs_all: Prediction probabilities (optional)
        n: Number of bootstrap iterations
        alpha: Confidence level
        
    Returns:
        mean_dict: Dictionary with mean values for each metric
        std_dict: Dictionary with standard deviation values for each metric
        all_scores: Dictionary with lists of all bootstrap scores for each metric
    """
    all_scores = {
        'balanced_acc': [],
        'kappa': [],
        'weighted_f1': []
    }
    
    num_classes = len(np.unique(targets_all))
    
    # Calculate metrics for all bootstrapped samples
    for seed in tqdm(range(n), desc="Bootstrap iterations"):
        np.random.seed(seed)
        # Sample with replacement
        bootstrap_ind = np.random.choice(
            len(targets_all), size=len(targets_all), replace=True
        )
        
        # Check if all classes are represented in the bootstrap sample
        # If not, resample
        collision = 0
        while len(np.unique(targets_all[bootstrap_ind])) != num_classes:
            np.random.seed(seed + collision + n)
            bootstrap_ind = np.random.choice(
                len(targets_all), size=len(targets_all), replace=True
            )
            collision += 1
            if collision % 100 == 0:
                print(f"Needed {collision} resamples to include all classes")
        
        # Get bootstrapped samples
        sample_targets = targets_all[bootstrap_ind]
        sample_preds = preds_all[bootstrap_ind]
        
        # Calculate metrics for this bootstrap sample
        all_scores['balanced_acc'].append(
            balanced_accuracy_score(sample_targets, sample_preds)
        )
        all_scores['kappa'].append(
            cohen_kappa_score(sample_targets, sample_preds)
        )
        all_scores['weighted_f1'].append(
            f1_score(sample_targets, sample_preds, average='weighted')
        )
    
    # Calculate mean and standard deviation for each metric
    mean_dict = {}
    std_dict = {}
    
    for metric in all_scores.keys():
        metric_values = np.array(all_scores[metric])
        mean_dict[metric] = np.mean(metric_values)
        std_dict[metric] = np.std(metric_values)
    
    return mean_dict, std_dict, all_scores


def evaluate_ebrains(train_data, val_data, test_data, max_iter=2000,
                    save_results=True, output_dir=None, model_name=None):
    """Evaluate model using balanced accuracy and Cohen's kappa"""
    # Prepare data
    X_train = np.stack(train_data['features'].values)
    y_train = train_data['label'].values
    X_val = np.stack(val_data['features'].values)
    y_val = val_data['label'].values
    X_test = np.stack(test_data['features'].values)
    y_test = test_data['label'].values

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Find best model
    best_score = -float('inf')
    best_model = None
    C_values = np.logspace(-100, 100, num=50)

    for C in tqdm(C_values, desc="Finding best C"):
        model = LogisticRegression(
            C=1/C,
            fit_intercept=True,
            class_weight='balanced',
            max_iter=max_iter,
            random_state=42,
            solver="lbfgs",
            tol=1e-5
        )
        model.fit(X_train, y_train)
        val_preds = model.predict(X_val)
        score = balanced_accuracy_score(y_val, val_preds)
        
        if score > best_score:
            best_score = score
            best_model = model

    # Evaluate final model
    test_preds = best_model.predict(X_test)
    test_proba = best_model.predict_proba(X_test)
    
    metrics = {
        'balanced_acc': balanced_accuracy_score(y_test, test_preds),
        'kappa': cohen_kappa_score(y_test, test_preds),
        'confusion_matrix': confusion_matrix(y_test, test_preds),
        'weighted_f1': f1_score(y_test, test_preds, average='weighted')
    }

    # Print results
    print("\nTest Results:")
    print(f"Balanced Accuracy: {metrics['balanced_acc']:.4f}")
    print(f"Cohen's Kappa: {metrics['kappa']:.4f}")
    print(f"Weighted F1 Score: {metrics['weighted_f1']:.4f}")
    
    # Calculate bootstrap confidence intervals
    print("\nCalculating bootstrap confidence intervals...")
    mean_metrics, std_metrics, all_bootstrap_scores = bootstrap(
        y_test, test_preds, test_proba, n=1000
    )
    print("\nBootstrap Results (mean ± std):")
    print(f"Balanced Accuracy: {mean_metrics['balanced_acc']:.4f} ± "
          f"{std_metrics['balanced_acc']:.4f}")
    print(f"Cohen's Kappa: {mean_metrics['kappa']:.4f} ± "
          f"{std_metrics['kappa']:.4f}")
    print(f"Weighted F1 Score: {mean_metrics['weighted_f1']:.4f} ± "
          f"{std_metrics['weighted_f1']:.4f}")
    
    # Save prediction results to CSV
    if save_results:
        if output_dir is None:
            raw_path = os.getcwd()
            output_dir = os.path.join(raw_path, 'results_fine')
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a DataFrame with slide_id, true label, predicted label, and probabilities
        results_df = pd.DataFrame({
            'slide_id': test_data['slide_id'].values,
            'diagnosis': test_data['diagnosis'].values,
            'true_label': y_test,
            'predicted_label': test_preds
        })
        
        # Add probability columns for each class
        for i in range(test_proba.shape[1]):
            class_name = [k for k, v in ebrains_label_dict.items() if v == i][0]
            results_df[f'prob_{i}_{class_name}'] = test_proba[:, i]
        
        # Add file name based on model name
        file_name = f"prediction_results_{model_name if model_name else 'model'}.csv"
        results_df.to_csv(os.path.join(output_dir, file_name), index=False)
        print(f"Prediction results saved to "
              f"{os.path.join(output_dir, file_name)}")
        
        # Save bootstrap results
        bootstrap_df = pd.DataFrame({
            'metric': ['balanced_acc', 'kappa', 'weighted_f1'],
            'mean': [mean_metrics['balanced_acc'], mean_metrics['kappa'],
                    mean_metrics['weighted_f1']],
            'std': [std_metrics['balanced_acc'], std_metrics['kappa'],
                   std_metrics['weighted_f1']]
        })
        bootstrap_file = f"bootstrap_results_{model_name if model_name else 'model'}.csv"
        bootstrap_df.to_csv(os.path.join(output_dir, bootstrap_file), index=False)
        print(f"Bootstrap results saved to "
              f"{os.path.join(output_dir, bootstrap_file)}")

    return metrics, (mean_metrics, std_metrics, all_bootstrap_scores)


def main(model_name="titan"):
    """Main function to run evaluation"""
    base_path = os.getcwd()
    feature_dir = os.path.join(base_path, model_name)
    output_dir = os.path.join(base_path, 'results_fine')
    
    print(f"\nEvaluating {model_name}")
    train_data, val_data, test_data = load_ebrains_data(feature_dir)
    results, bootstrap_results = evaluate_ebrains(
        train_data, val_data, test_data, save_results=True,
        output_dir=output_dir, model_name=model_name
    )
    
    return results, bootstrap_results


def evaluate_ensemble(model_names=["titan"]):
    """Evaluate ensemble of multiple models by averaging their features"""
    base_path = os.getcwd()
    feature_dirs = [os.path.join(base_path, model_name) for model_name in model_names]
    output_dir = os.path.join(base_path, 'results_fine')
    ensemble_name = "_".join(model_names)
    
    # Load splits
    train_df = pd.read_csv(os.path.join(base_path, "train.csv"))
    val_df = pd.read_csv(os.path.join(base_path, "val.csv"))
    test_df = pd.read_csv(os.path.join(base_path, "test.csv"))
    
    # Load and average features from multiple models
    features_dict = {}
    slide_ids = (train_df['slide_id'].tolist() + val_df['slide_id'].tolist() +
                 test_df['slide_id'].tolist())
    
    for slide_id in tqdm(slide_ids, desc="Loading features"):
        features_list = []
        for feature_dir in feature_dirs:
            file_path = os.path.join(feature_dir, f"{slide_id}.h5")
            if os.path.exists(file_path):
                with h5py.File(file_path, 'r') as f:
                    if any(model in feature_dir for model in [
                        "titan", "prov_gigapath", "chief"]):
                        feat = f["features"][:][0]
                        feat = (feat - np.mean(feat)) / np.std(feat)
                        features_list.append(feat)
                    else:
                        feat = f["features_dim"][:][0]
                        feat = (feat - np.mean(feat)) / np.std(feat)
                        features_list.append(feat)
        if features_list:
            features_dict[slide_id] = np.concatenate(features_list, axis=0)

    # Add features and labels
    for df in [train_df, val_df, test_df]:
        df['features'] = df['slide_id'].map(features_dict)
        df['label'] = df['diagnosis'].map(ebrains_label_dict)
        df = df.dropna(subset=['features'])

    print(f"\nEvaluating ensemble of models: {', '.join(model_names)}")
    print(f"Training set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")
    print(f"Test set size: {len(test_df)}")
    print(f"Total set size: {len(train_df) + len(val_df) + len(test_df)}")

    results, bootstrap_results = evaluate_ebrains(
        train_df, val_df, test_df, save_results=True,
        output_dir=output_dir, model_name=f"ensemble_elf"
    )
    return results, bootstrap_results


if __name__ == "__main__":
    # Example usage:
    # Single model evaluation
    # results, bootstrap_results = main(model_name="uni_elf")
    # results, bootstrap_results = main(model_name="prov_gigapath")
    
    # Evaluate multiple models and visualize bootstrap comparison
    models_to_compare = [
        "chief", 
        "prov_gigapath",
        "titan"
    ]
    
    all_results = []
    all_bootstrap_results = []
    
    for model in models_to_compare:
        print(f"\nEvaluating {model}")
        results, bootstrap_res = main(model_name=model)
        all_results.append(results)
        all_bootstrap_results.append(bootstrap_res)
    
    # Ensemble evaluation
    ensemble_name = "elf"
    print(f"\nEvaluating {ensemble_name}")
    ensemble_results, ensemble_bootstrap = evaluate_ensemble(model_names=[
        "virchow2_elf", 
        "conch_v1_5_elf", 
        "h0_elf", 
        "gigapath_elf", 
        "uni_elf"
    ])
    