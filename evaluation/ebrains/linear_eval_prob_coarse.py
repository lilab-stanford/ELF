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


ebrains_label_dict = {
    'Adult-type diffuse gliomas': 0,
    'Meningiomas': 1,
    'Tumours of the sellar region': 2,
    'Mesenchymal, non-meningothelial tumours involving the CNS': 3,
    'Glioneuronal and neuronal tumours': 4,
    'Cranial and paraspinal nerve tumours': 5,
    'Circumscribed astrocytic gliomas ': 6,
    'Ependymal Tumours': 7,
    'Metastatic tumours': 8,
    'Haematolymphoid tumours involving the CNS': 9,
    'Paediatric-type diffuse low-grade gliomas ': 10,
    'Embryonal Tumors': 11
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
                    "titan", "prov_gigapath", "chief"
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
        df['label'] = df['diagnosis_group'].map(ebrains_label_dict)
        df = df.dropna(subset=['features'])

    # Print the size of the dataset
    print(f"Train set size: {len(train_df)}")
    print(f"Val set size: {len(val_df)}")
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
            C=C,
            fit_intercept=True,
            class_weight='balanced',
            max_iter=max_iter,
            random_state=42,
            solver="lbfgs",
            tol=1e-4
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
            base_path = os.getcwd()
            output_dir = os.path.join(base_path, 'results_coarse')
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a DataFrame with slide_id, true label, predicted label, and probabilities
        results_df = pd.DataFrame({
            'slide_id': test_data['slide_id'].values,
            'diagnosis_group': test_data['diagnosis_group'].values,
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
        
        # Save confusion matrix as an image
        plt.figure(figsize=(10, 8))
        cm = metrics['confusion_matrix']
        
        # Convert class indices to class names for better visualization
        class_names = [k for k, v in sorted(ebrains_label_dict.items(),
                                           key=lambda item: item[1])]
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot confusion matrix
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Normalized Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save confusion matrix
        confusion_file = f"confusion_matrix_{model_name if model_name else 'model'}.png"
        plt.savefig(os.path.join(output_dir, confusion_file), dpi=300,
                   bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix saved to "
              f"{os.path.join(output_dir, confusion_file)}")

    return metrics, (mean_metrics, std_metrics, all_bootstrap_scores)


def main(model_name="titan"):
    """Main function to run evaluation"""
    base_path = os.getcwd()
    feature_dir = os.path.join(base_path, model_name)
    output_dir = os.path.join(base_path, 'results_coarse')
    
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
    output_dir = os.path.join(base_path, 'results_coarse')
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
                    if "virchow2" in file_path:
                        feat = f["features_dim"][:][0]
                        feat = (feat - np.mean(feat)) / np.std(feat)
                        features_list.append(feat)
                    elif any(model in file_path for model in [
                        "titan", "prov_gigapath", "chief"
                    ]):
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
        df['label'] = df['diagnosis_group'].map(ebrains_label_dict)
        df = df.dropna(subset=['features'])

    print(f"\nEvaluating ensemble of models: {', '.join(model_names)}")
    print(f"Train set size: {len(train_df)}")
    print(f"Val set size: {len(val_df)}")
    print(f"Test set size: {len(test_df)}")
    print(f"Total set size: {len(train_df) + len(val_df) + len(test_df)}")

    results, bootstrap_results = evaluate_ebrains(
        train_df, val_df, test_df, save_results=True,
        output_dir=output_dir, model_name=f"ensemble_{ensemble_name}"
    )
    return results, bootstrap_results


if __name__ == "__main__":
    models_to_compare = [
        "prov_gigapath",
        "titan",
        "chief"
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