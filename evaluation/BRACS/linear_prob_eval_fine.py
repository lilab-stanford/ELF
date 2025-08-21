import os
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import resample
from scipy import stats

bracs_label_dict = {
    'N': 0,     # Normal
    'PB': 1,    # Proliferative Breast
    'UDH': 2,   # Usual Ductal Hyperplasia
    'ADH': 3,   # Atypical Ductal Hyperplasia
    'FEA': 4,   # Flat Epithelial Atypia
    'DCIS': 5,  # Ductal Carcinoma In Situ
    'IC': 6     # Invasive Carcinoma
}


def load_bracs_data(feature_dir, csv_file, balance_method=None):
    """Load BRACS dataset including features and labels"""
    # Load CSV file
    df = pd.read_csv(csv_file)
    
    # Confirm required columns exist
    required_columns = ['WSI Filename', 'WSI label', 'Set']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"CSV file missing required column: {col}")

    # Rename columns for easier use
    df = df.rename(columns={
        'WSI Filename': 'slide_id',
        'WSI label': 'diagnosis',
        'Set': 'set'
    })
    
    # Split dataset
    train_df = df[df['set'] == 'Training'].copy()
    val_df = df[df['set'] == 'Validation'].copy()
    test_df = df[df['set'] == 'Testing'].copy()
    
    # If no validation set, split 10% from training set as validation set
    if len(val_df) == 0:
        from sklearn.model_selection import train_test_split
        train_df, val_df = train_test_split(
            train_df, test_size=0.1, random_state=42, stratify=train_df['diagnosis']
        )
    
    # Load features
    features_dict = {}
    feature_files = [f for f in os.listdir(feature_dir) if f.endswith('.h5')]
    
    for file_name in tqdm(feature_files, desc="Reading feature files"):
        slide_id = file_name.replace('.h5', '')
        try:
            with h5py.File(os.path.join(feature_dir, file_name), 'r') as f:
                if ("titan" in feature_dir or "prov_gigapath" in feature_dir or 
                    "chief" in feature_dir):
                    feat = f["features"][:][0]
                else:
                    feat = f["features_dim"][:][0]
                
                features_dict[slide_id] = feat
        except Exception as e:
            print(f"Cannot load feature file {file_name}: {e}")
    
    # Add features and labels
    for df in [train_df, val_df, test_df]:
        # Ensure slide_id format matches feature filename
        df['features'] = df['slide_id'].map(features_dict)
        df['label'] = df['diagnosis'].map(bracs_label_dict)
    
    # Check feature matching
    for dataset_name, dataset in zip(['Training set', 'Validation set', 'Test set'], 
                                    [train_df, val_df, test_df]):
        missing_features = dataset['features'].isna().sum()
        if missing_features > 0:
            print(f"{dataset_name} has {missing_features} samples missing features")
            print(f"Samples missing features: {dataset[dataset['features'].isna()]['slide_id'].tolist()[:5]}...")
        
        # Remove samples missing features
        dataset.dropna(subset=['features'], inplace=True)
    
    # Apply balanced sampling to training set only
    if balance_method:
        train_df = balance_dataset(train_df, method=balance_method)
    
    return train_df, val_df, test_df


def bootstrap(targets_all, preds_all, probs_all=None, n=1000, alpha=0.95):
    """
    Calculate evaluation metrics using bootstrap method.
    
    Parameters:
        targets_all: true labels
        preds_all: predicted labels
        probs_all: prediction probabilities (optional)
        n: number of bootstrap iterations
        alpha: confidence level
        
    Returns:
        mean_dict: dictionary of metric means
        std_dict: dictionary of metric standard deviations
        all_scores: dictionary of all bootstrap scores
    """
    all_scores = {
        'balanced_acc': [],
        'weighted_f1': []
    }
    
    num_classes = len(np.unique(targets_all))
    
    # Calculate metrics for all bootstrap samples
    for seed in tqdm(range(n), desc="Bootstrap iterations"):
        np.random.seed(seed)
        # Sample with replacement
        bootstrap_ind = np.random.choice(
            len(targets_all), size=len(targets_all), replace=True
        )
        
        # Check if bootstrap sample contains all classes
        # If not, resample
        collision = 0
        while len(np.unique(targets_all[bootstrap_ind])) != num_classes:
            np.random.seed(seed + collision + n)
            bootstrap_ind = np.random.choice(
                len(targets_all), size=len(targets_all), replace=True
            )
            collision += 1
            if collision % 100 == 0:
                print(f"Need {collision} resamples to include all classes")
        
        # Get bootstrap sample
        sample_targets = targets_all[bootstrap_ind]
        sample_preds = preds_all[bootstrap_ind]
        
        # Calculate metrics for this bootstrap sample
        all_scores['balanced_acc'].append(
            balanced_accuracy_score(sample_targets, sample_preds)
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


def evaluate_bracs(train_data, val_data, test_data, max_iter=2000, save_results=True, 
                   output_dir=None, model_name=None, types=None):
    """Evaluate model using balanced accuracy and weighted F1 score"""
    # Prepare data - correctly convert from Series to 2D array
    X_train = np.stack(train_data['features'].values)
    y_train = train_data['label'].values
    X_val = np.stack(val_data['features'].values)
    y_val = val_data['label'].values
    X_test = np.stack(test_data['features'].values)
    y_test = test_data['label'].values
    
    # Feature standardization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Find best model
    best_score = -float('inf')
    best_model = None
    C_values = np.logspace(-100, 100, num=50)

    for C in tqdm(C_values, desc="Finding best regularization parameter"):
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

    # Evaluate best model on validation set
    val_preds = best_model.predict(X_val)
    val_metrics = {
        'balanced_acc': balanced_accuracy_score(y_val, val_preds),
        'weighted_f1': f1_score(y_val, val_preds, average='weighted')
    }

    # Evaluate final model on test set
    test_preds = best_model.predict(X_test)
    test_proba = best_model.predict_proba(X_test)
    
    test_metrics = {
        'balanced_acc': balanced_accuracy_score(y_test, test_preds),
        'weighted_f1': f1_score(y_test, test_preds, average='weighted'),
        'confusion_matrix': confusion_matrix(y_test, test_preds)
    }

    # Calculate bootstrap confidence intervals
    mean_metrics, std_metrics, all_bootstrap_scores = bootstrap(
        y_test, test_preds, test_proba, n=1000
    )
    
    # Print only the best results
    print(f"\n{model_name} - Best Results:")
    print(f"Balanced Accuracy: {mean_metrics['balanced_acc']:.4f} ± {std_metrics['balanced_acc']:.4f}")
    print(f"Weighted F1: {mean_metrics['weighted_f1']:.4f} ± {std_metrics['weighted_f1']:.4f}")
    
    # Save prediction results to CSV
    if save_results:
        if output_dir is None:
            base_path = os.getcwd()
            output_dir = os.path.join(base_path, 'results')
        os.makedirs(output_dir, exist_ok=True)
        
        results_df = pd.DataFrame({
            'slide_id': test_data['slide_id'].values,
            'diagnosis': test_data['diagnosis'].values,
            'true_label': y_test,
            'predicted_label': test_preds
        })
        
        # Add probability columns for each class
        for i in range(test_proba.shape[1]):
            class_name = [k for k, v in bracs_label_dict.items() if v == i][0]
            results_df[f'prob_{i}_{class_name}'] = test_proba[:, i]
        
        # Add filename based on model name
        file_name = f"prediction_results_{model_name if model_name else 'model'}_{types}.csv"
        results_df.to_csv(os.path.join(output_dir, file_name), index=False)
        
        # Save bootstrap results
        bootstrap_df = pd.DataFrame({
            'metric': ['balanced_acc', 'weighted_f1'],
            'mean': [mean_metrics['balanced_acc'], mean_metrics['weighted_f1']],
            'std': [std_metrics['balanced_acc'], std_metrics['weighted_f1']]
        })
        bootstrap_file = f"bootstrap_results_{model_name if model_name else 'model'}.csv"
        bootstrap_df.to_csv(os.path.join(output_dir, bootstrap_file), index=False)
        
    return test_metrics, (mean_metrics, std_metrics, all_bootstrap_scores)


def balance_dataset(df, label_col='label', method='downsample', random_state=42):
    """
    Balance dataset using sampling
    
    Parameters:
    df (pandas.DataFrame): dataset containing features and labels
    label_col (str): name of label column
    method (str): 'downsample' for downsampling, 'upsample' for upsampling
    random_state (int): random seed
    
    Returns:
    pandas.DataFrame: balanced dataset
    """
    # Get class counts
    class_counts = df[label_col].value_counts()
    
    if method == 'downsample':
        # Downsample to minimum class count
        min_class_count = class_counts.min()
        balanced_dfs = []
        
        # Downsample each class
        for class_label in class_counts.index:
            class_df = df[df[label_col] == class_label]
            # If count exceeds minimum class count, downsample
            if len(class_df) > min_class_count:
                downsampled = resample(
                    class_df, 
                    replace=False,  # Sample without replacement
                    n_samples=min_class_count,
                    random_state=random_state
                )
                balanced_dfs.append(downsampled)
            else:
                balanced_dfs.append(class_df)
                
        # Combine all downsampled data
        balanced_df = pd.concat(balanced_dfs)
        
    elif method == 'upsample':
        # Upsample to maximum class count
        max_class_count = class_counts.max()
        balanced_dfs = []
        
        # Upsample each class
        for class_label in class_counts.index:
            class_df = df[df[label_col] == class_label]
            # If count is less than maximum class count, upsample
            if len(class_df) < max_class_count:
                upsampled = resample(
                    class_df, 
                    replace=True,  # Sample with replacement
                    n_samples=max_class_count,
                    random_state=random_state
                )
                balanced_dfs.append(upsampled)
            else:
                balanced_dfs.append(class_df)
                
        # Combine all upsampled data
        balanced_df = pd.concat(balanced_dfs)
    
    else:
        raise ValueError("method must be 'downsample' or 'upsample'")
    
    # Reshuffle data
    balanced_df = balanced_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    return balanced_df


def evaluate_ensemble(model_names, balance_method=None, types="elf"):
    """Evaluate ensemble model by combining features from multiple models"""
    base_path = os.getcwd()
    feature_dirs = [os.path.join(base_path, model_name) for model_name in model_names]
    csv_file = os.path.join(base_path, "BRACS.csv")
    output_dir = os.path.join(base_path, 'results')
    ensemble_name = "_".join([m.split("_")[0] for m in model_names])
    
    # Load CSV file
    df = pd.read_csv(csv_file)
    
    # Rename columns
    df = df.rename(columns={
        'WSI Filename': 'slide_id',
        'WSI label': 'diagnosis',
        'Set': 'set'
    })
    
    # Split dataset
    train_df = df[df['set'] == 'Training'].copy()
    val_df = df[df['set'] == 'Validation'].copy()
    test_df = df[df['set'] == 'Testing'].copy()
    
    # If no validation set, split 10% from training set as validation set
    if len(val_df) == 0:
        from sklearn.model_selection import train_test_split
        train_df, val_df = train_test_split(
            train_df, test_size=0.1, random_state=42, stratify=train_df['diagnosis']
        )
    
    # Load and combine features from multiple models
    features_dict = {}
    slide_ids = train_df['slide_id'].tolist() + val_df['slide_id'].tolist() + test_df['slide_id'].tolist()
    
    for slide_id in tqdm(slide_ids, desc="Loading ensemble features"):
        features_list = []
        for feature_dir in feature_dirs:
            file_path = os.path.join(feature_dir, f"{slide_id}.h5")
            if os.path.exists(file_path):
                with h5py.File(file_path, 'r') as f:
                    if ("titan" in feature_dir or "prov_gigapath" in feature_dir or 
                        "chief" in feature_dir):
                        feat = f["features"][:][0]
                    else:
                        feat = f["features_dim"][:][0]
                    
                    features_list.append(feat)
        
        if features_list:
            features_dict[slide_id] = np.concatenate(features_list, axis=0)
            
    # Add features and labels
    for df in [train_df, val_df, test_df]:
        df['features'] = df['slide_id'].map(features_dict)
        df['label'] = df['diagnosis'].map(bracs_label_dict)
        df.dropna(subset=['features'], inplace=True)
    
    # Apply balanced sampling to training set
    if balance_method:
        train_df = balance_dataset(train_df, method=balance_method)

    results, bootstrap_results = evaluate_bracs(
        train_df, val_df, test_df, 
        save_results=True, 
        output_dir=output_dir, 
        model_name=f"ensemble_{ensemble_name}",
        types=types
    )
    return results, bootstrap_results


def main(model_name="titan", balance_method=None):
    """Main function to run evaluation"""
    base_path = os.getcwd()
    csv_file = os.path.join(base_path, "BRACS.csv")
    output_dir = os.path.join(base_path, 'results')
    feature_dir = os.path.join(base_path, model_name)
    train_data, val_data, test_data = load_bracs_data(feature_dir, csv_file, balance_method=balance_method)
    results, bootstrap_results = evaluate_bracs(
        train_data, val_data, test_data, 
        save_results=True,
        output_dir=output_dir,
        model_name=model_name
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
    
    # Single model evaluation
    for model in models_to_compare:
        results, bootstrap_res = main(model_name=model, balance_method="upsample")
        all_results.append(results)
        all_bootstrap_results.append(bootstrap_res)
    
    # Ensemble model evaluation
    ensemble_name = "elf"
    print(f"\nEvaluating {ensemble_name}")
    ensemble_results, ensemble_bootstrap = evaluate_ensemble(
        model_names=[
            "virchow2_elf", 
            "conch_v1_5_elf", 
            "h0_elf", 
            "gigapath_elf", 
            "uni_elf"
        ],
        balance_method="upsample"
    )
    
    # Add ensemble to comparison
    models_to_compare.append(ensemble_name)
    all_bootstrap_results.append(ensemble_bootstrap)
    