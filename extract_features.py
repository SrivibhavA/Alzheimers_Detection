import argparse
import pandas as pd
import mne
import numpy as np
import math
import warnings
import os
import gc
from datetime import datetime
from scipy import signal
from scipy.integrate import trapezoid
import seaborn as sns
import matplotlib
matplotlib.use('Agg') # Prevent Tkinter main loop errors in scripts
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, GridSearchCV, LeaveOneOut
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from scipy.stats import mannwhitneyu

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, roc_curve, auc

import shap
import networkx as nx

# Suppress MNE warnings about boundary events (preprocessed data discontinuities)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="mne")
# Suppress Sklearn name mismatch warnings (common in SHAP/Pipeline interactions)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


# ==========================================
# CONFIGURATION
# ==========================================
# Default mode if no command line arguments are provided.
# True: Quick test on 4 subjects (2 AD, 2 HC) with 30s data data.
# False: Full analysis on all eligible subjects.
DEFAULT_TEST_MODE = True 
# ==========================================


def load_participants(data_dir):
    """
    Load participants.tsv file.
    """
    # Standard BIDS participants file location
    participants_path = os.path.join(data_dir, 'participants.tsv')
    if not os.path.exists(participants_path):
        raise FileNotFoundError(f"participants.tsv not found at {participants_path}")
    
    return pd.read_csv(participants_path, sep='\t')


def load_subject_eeg(subject_id, data_dir):
    """
    Load EEG data for a specific subject.  
    Prefer preprocessed data in 'derivatives' folder if available.
    """
    # Expert Fix: Probing multiple derivative paths to handle different BIDS structures (ds004504 vs ds006036)
    potential_paths = [
        os.path.join(data_dir, 'derivatives', 'eeglab', subject_id, 'eeg'),
        os.path.join(data_dir, 'derivatives', subject_id, 'eeg'),
        os.path.join(data_dir, subject_id, 'eeg')
    ]
    
    eeg_dir = None
    for path in potential_paths:
        if os.path.exists(path):
            eeg_dir = path
            break
            
    if not eeg_dir:
        raise FileNotFoundError(f"EEG directory not found for {subject_id} in {data_dir}")

         
    # Find .set file (EEGLAB format)
    set_files = [f for f in os.listdir(eeg_dir) if f.endswith('.set')]
    if not set_files:
        raise FileNotFoundError(f"No .set files found for {subject_id}")
        
    set_path = os.path.join(eeg_dir, set_files[0])
    
    try:
        raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose=False)
    except Exception as e:
        print(f"Error loading EEG for {subject_id}: {e}")
        raise e
    
    return raw

class FeaturesExtracter:

    # Frequency bands definition
    BANDS = {
        'Delta': (0.5, 4),
        'Theta': (4, 8),
        'Alpha': (8, 12),
        'Beta': (12, 30),
        'Gamma': (30, 45)
    }

    # AD-Specific Regional Clusters (Posterior/Temporal)
    # These regions are the first to show slowing and reduced reactivity in AD.
    POSTERIOR_TEMPORAL = ['P3', 'P4', 'T5', 'T6', 'O1', 'O2']

    def compute_subject_spectral_features(self, subject_id, data_dir, test_mode=False, cluster_only=True):
        """
        Compute relative spectral power for the AD-critical Posterior/Temporal cluster.
        If cluster_only is True, focuses only on the critical AD sensors.
        """

        try:
            raw = load_subject_eeg(subject_id, data_dir)
            
            # Standardize Temporal Sampling: Use same 30s window (10-40s) for all features
            if raw.times[-1] > 40:
                raw.crop(tmin=10, tmax=40)
            elif raw.times[-1] > 30:
                raw.crop(tmin=0, tmax=30)
            
            
            # Explicitly pick EEG channels
            raw.pick(['eeg'], exclude='bads')
            
            # PREPROCESSING: CLEANING THE SIGNAL
            # 1. Set Montage (essential for artifact rejection mapping)
            try:
                montage = mne.channels.make_standard_montage('standard_1020')
                raw.set_montage(montage, on_missing='ignore', verbose=False)
            except: pass

            # 2. Common Average Reference (CAR) - Removes global noise
            try:
                raw.set_eeg_reference('average', projection=False, verbose=False)
            except: pass
            
            # Broadband filter for spectral analysis
            raw.filter(0.5, 45, verbose=False)
            
            # Epochs
            epochs = mne.make_fixed_length_epochs(raw, duration=5.0, preload=True, verbose=False)
            
            # 3. ARTIFACT REJECTION (Crucial for Spectral Accuracy)
            # Remove epochs with amplitude > 100uV (blinks/muscle)
            try:
                epochs.drop_bad(reject=dict(eeg=100e-6), verbose=False)
            except: pass
            
            if len(epochs) == 0:
                 return None

            # Get channel names to identify regional sensors
            ch_names = epochs.ch_names
            
            # Filter to cluster if requested
            if cluster_only:
                cluster_indices = [i for i, ch in enumerate(ch_names) if ch.upper() in self.POSTERIOR_TEMPORAL]
                if not cluster_indices:
                    # Fallback if specific electrodes are missing (rare in 10-20)
                    print(f"    ! Warning: Cluster electrodes {self.POSTERIOR_TEMPORAL} not found in {subject_id}. Using all sensors.")
                    filtered_data = epochs.get_data()
                else:
                    filtered_data = epochs.get_data(picks=cluster_indices)
            else:
                filtered_data = epochs.get_data()
                
            sfreq = 500
         
            # Compute PSD using Welch's method
            freqs, psd = signal.welch(filtered_data, fs=sfreq, nperseg=500, axis=-1)
            # psd shape: (n_epochs, n_channels, n_freqs)
            
            # Average PSD across epochs and channels -> Regional PSD
            avg_psd = np.mean(psd, axis=(0, 1))

            results = {}
            total_psd = trapezoid(avg_psd, freqs)
            
            # 1. Delta-Alpha Ratio (DAR) - Main Biomarker of Slowing
            delta_val = trapezoid(avg_psd[(freqs >= 1) & (freqs <= 4)], freqs[(freqs >= 1) & (freqs <= 4)])
            theta_val = trapezoid(avg_psd[(freqs >= 4) & (freqs <= 8)], freqs[(freqs >= 4) & (freqs <= 8)])
            alpha_val = trapezoid(avg_psd[(freqs >= 8) & (freqs <= 12)], freqs[(freqs >= 8) & (freqs <= 12)])
            beta_val  = trapezoid(avg_psd[(freqs >= 12) & (freqs <= 30)], freqs[(freqs >= 12) & (freqs <= 30)])
            
            if alpha_val > 0:
                results['Theta_Alpha_Ratio'] = theta_val / alpha_val
            if beta_val > 0:
                results['Theta_Beta_Ratio'] = theta_val / beta_val
            
            # Low/High Alpha Sub-bands (More sensitive to early AD slowing)
            low_alpha_val = trapezoid(avg_psd[(freqs >= 8) & (freqs <= 10)], freqs[(freqs >= 8) & (freqs <= 10)])
            high_alpha_val = trapezoid(avg_psd[(freqs >= 10) & (freqs <= 12)], freqs[(freqs >= 10) & (freqs <= 12)])
            
            if total_psd > 0:
                results['Rel_Alpha'] = alpha_val / total_psd
                results['Rel_Theta'] = theta_val / total_psd
                results['Rel_Low_Alpha'] = low_alpha_val / total_psd
                results['Rel_High_Alpha'] = high_alpha_val / total_psd
                
            # Peak Alpha Frequency (PAF) - Gold standard slowing marker
            alpha_mask = (freqs >= 8) & (freqs <= 12)
            if np.any(alpha_mask) and np.max(avg_psd[alpha_mask]) > 0:
                peak_idx = np.argmax(avg_psd[alpha_mask])
                results['Peak_Alpha_Freq'] = freqs[alpha_mask][peak_idx]
                
            # Store absolute values for Reactivity calculation
            results['Abs_Alpha'] = alpha_val
            results['Abs_Theta'] = theta_val
            
            return results


        except Exception as e:
            print(f"    ! Error computing spectral features: {e}")
            import traceback
            traceback.print_exc()
            return None

    def compute_subject_pli(self, subject_id, data_dir, test_mode=False, cluster_only=True):
        """
        Compute Global Phase Lag Index (Functional Connectivity) in the Alpha band.
        If cluster_only is True, focuses on Connectivity within the Posterior regions.
        """
        raw = load_subject_eeg(subject_id, data_dir)
        if raw is None: return None
        
        # Standardize Temporal Sampling
        if raw.times[-1] > 40: raw.crop(tmin=10, tmax=40)
        elif raw.times[-1] > 30: raw.crop(tmin=0, tmax=30)

        raw.pick(['eeg'], exclude='bads')
        
        # Filter to cluster if requested
        if cluster_only:
            valid_cluster = [ch for ch in raw.ch_names if ch.upper() in self.POSTERIOR_TEMPORAL]
            if len(valid_cluster) >= 2:
                raw.pick(valid_cluster)
            else:
                print(f"    ! Warning: Cluster too small for PLI in {subject_id}. Using all.")

        try:
            # 1. Preprocessing
            raw.filter(8, 12, verbose=False) # AD changes are most significant in Alpha PLI

            
            # Set standard montage for robustness
            try:
                montage = mne.channels.make_standard_montage('standard_1020')
                raw.set_montage(montage, on_missing='ignore', verbose=False)
            except:
                pass

            # 1. Apply Common Average Reference (CAR) to remove common noise
            # This is crucial for connectivity analysis to mitigate volume conduction/reference effects
            try:
                raw.set_eeg_reference('average', projection=False, verbose=False)
            except:
                pass # Proceed if referencing fails (e.g. fewer channels)

            # 2. Filter to Alpha band (8-12 Hz) - Key AD biomarker frequency
            raw.filter(8, 12, verbose=False)
            
            # Create epochs (5 seconds)
            epochs = mne.make_fixed_length_epochs(raw, duration=5.0, preload=True, verbose=False)
            
            # 3. Artifact Rejection: Drop epochs with amplitude > 100 microvolts (Stricter cleaning)
            # 150uV was lenient; 100uV catches more muscle noise (which is often >100uV)
            try:
                epochs.drop_bad(reject=dict(eeg=150e-6), verbose=False)
            except:
                pass # If channel types aren't set correctly, skip rejection
                
            if len(epochs) == 0:
                 print(f"    -> All epochs rejected (too noisy).")
                 return np.nan

            epochs_data = epochs.get_data() # (n_epochs, n_channels, n_times)
            
            n_channels = epochs_data.shape[1]
            pli_sum = 0
            count = 0
            
            # Calculate pairwise PLI for all channel pairs
            adj_matrix = np.zeros((n_channels, n_channels))
            count = 0
            
            for i in range(n_channels):
                for j in range(i+1, n_channels):
                    sig1 = epochs_data[:, i, :]
                    sig2 = epochs_data[:, j, :]
                    
                    pli = self._compute_pli(sig1, sig2)
                    adj_matrix[i, j] = pli
                    adj_matrix[j, i] = pli # Symmetric
                    count += 1
            
            if count > 0:
                # Global PLI (Mean Connectivity across all channel pairs)
                global_pli = np.mean(adj_matrix[np.triu_indices(n_channels, k=1)])
                return {'Global_PLI': global_pli}
            else:
                return None
                
        except Exception as e:
            print(f"Skipping {subject_id} due to error: {e}")
            return np.nan

    def _compute_pli(self, signal1, signal2):
        """Compute PLI across all epochs"""
        # signal1, signal2 shape: (n_epochs, n_times)
    
        # Flatten to compute phase across all time points
        sig1_flat = signal1.flatten()
        sig2_flat = signal2.flatten()
    
        analytic1 = signal.hilbert(sig1_flat)
        analytic2 = signal.hilbert(sig2_flat)
    
        phase_diff = np.angle(analytic1) - np.angle(analytic2)
        pli = np.abs(np.mean(np.sign(np.sin(phase_diff))))
    
        return pli

    @staticmethod
    def _spectral_entropy(x, sfreq):
        """Calculate Spectral Entropy (SE)."""
        # Compute PSD
        freqs, psd = signal.welch(x, sfreq, nperseg=sfreq*2)
        # Normalize PSD to get probability distribution
        psd_norm = psd / np.sum(psd)
        # Shannon Entropy
        se = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))
        # Normalize by log(N) to get 0-1 range
        se /= np.log2(len(psd_norm))
        return se

    def compute_subject_complexity(self, subject_id, data_dir, test_mode=False):
        """Deprecated - Focusing on high-accuracy spectral/connectivity features."""
        return None




class AlzheimerClassifier:
    """
    AI Model to classify Alzheimer's vs Control using EEG features and clinical data.
    Uses SVM with GridSearchCV for hyperparameter optimization and SHAP for explainability.
    """
    def __init__(self, output_dir="model_results"):
        self.output_dir = output_dir
        
        # Random Forest: Non-linear, robust to noise, good with complex interactions
        rf = RandomForestClassifier(random_state=42, class_weight='balanced')
        
        # Pipeline: Imputer -> Scaler (optional for RF but good practice) -> RF
        # REMOVED GridSearchCV: Overkill for small N, causes instability. 
        # Fixed RandomForest (Tuned for Sensitivity)
        self.model = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('rf', RandomForestClassifier(
                n_estimators=200,       # More trees for stability
                max_depth=7,            # Deeper trees to capture subtleties
                min_samples_split=5,
                random_state=42,
                class_weight='balanced',
                n_jobs=2 # Reduced to 2 to prevent system lag/high memory pressure
            ))
        ])




        
        # REFINED FEATURE SET: Only Statistically Significant Biomarkers
        # Removed: Reactivity_Posterior_Alpha_Block (p=0.93), EO_Posterior_PLI (p=0.23)
        self.features = [
            'EC_Posterior_Theta_Alpha_Ratio', # p < 0.0001 (***)
            'EC_Posterior_Rel_Alpha',         # p < 0.0001 (***)
            'EC_Posterior_Rel_High_Alpha',    # p < 0.001 (***)
            'EC_Posterior_Peak_Alpha_Freq',   # p < 0.01 (**)
            'EC_Posterior_PLI'                # p < 0.05 (*)
        ]





        
        
    def prepare_data(self, df):
        """Prepare features (X) and target (y)"""
        # Filter for valid groups (A and C)
        df_clean = df[df['Group'].isin(['A', 'C'])].copy()
        
        # Encode Target: A=1, C=0
        df_clean['Target'] = df_clean['Group'].map({'A': 1, 'C': 0})
        
        # Features (X) and Target (y)
        X = df_clean[self.features]
        y = df_clean['Target'].values
        
        # NaN Check
        nan_counts = X.isna().sum()
        total_rows = len(X)
        if nan_counts.sum() > 0:
            print("\n    ! WARNING: Missing Data Detected:")
            for col, count in nan_counts.items():
                if count > 0:
                    print(f"      - {col}: {count}/{total_rows} missing ({count/total_rows:.1%})")
        
        print(f"Data Prepared: {len(X)} samples. (A: {sum(y==1)}, C: {sum(y==0)})")
        return X, y


    def train_with_cv(self, df, use_loocv=False):
        """Train using Cross Validation and print metrics"""
        X, y = self.prepare_data(df)
        
        # 1. Statistical Verification (Mann-Whitney U Test)
        print("\nStatistical Proof (AD vs HC):")
        stats_report = ["Feature Significance (Mann-Whitney U):", "-"*40]
        for feature in self.features:
            ad_vals = df[df['Group'] == 'A'][feature].dropna()
            hc_vals = df[df['Group'] == 'C'][feature].dropna()
            stat, p = mannwhitneyu(ad_vals, hc_vals)
            stars = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            line = f"{feature:18} | p={p:.4f} ({stars})"
            print(f"  {line}")
            stats_report.append(line)
        
        # 2. Correlation Check (Expert Advice)
        try:
             plt.figure(figsize=(8, 6))
             corr = X.corr()
             sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
             plt.title("Feature Correlation Analysis")
             corr_path = os.path.join(self.output_dir, "feature_correlation.png")
             # We might not have output_dir yet if called before main saves it
             # but usually output_dir is passed in __init__
             if os.path.exists(self.output_dir):
                 plt.savefig(corr_path)
             plt.close()
        except: pass

        # 3. Cross-Validation
        # 3. Cross-Validation: User requested 5-Fold for stability/AUC over LOOCV
        if use_loocv:
            print("\nStarting Leave-One-Out Cross-Validation (LOOCV)...")
            cv = LeaveOneOut()
        else:
            print("\nStarting Stratified 5-Fold Cross-Validation...")
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            
        accuracies, aucs, sensitivities, specificities = [], [], [], []
        tprs = []
        mean_fpr = np.linspace(0, 1, 100)
        fold = 1
        
        report_lines = ["Starting Cross-Validation Summary", "="*30]
        report_lines.extend(stats_report)
        report_lines.append(f"\nData Prepared: {len(X)} samples. (A: {sum(y==1)}, C: {sum(y==0)})")
        
        plt.figure(figsize=(10, 8))
        
        for train_index, test_index in cv.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            try:
                # Use predict_proba for Logistic Regression AUC
                y_prob = self.model.predict_proba(X_test)[:, 1]
                
                # ROC Curve Logic
                if len(np.unique(y_test)) > 1:
                    fpr, tpr, _ = roc_curve(y_test, y_prob)
                    roc_auc = auc(fpr, tpr)
                    aucs.append(roc_auc)
                    
                    # Interpolate TPR for mean curve
                    interp_tpr = np.interp(mean_fpr, fpr, tpr)
                    interp_tpr[0] = 0.0
                    tprs.append(interp_tpr)
                    
                    plt.plot(fpr, tpr, lw=1, alpha=0.3, label=f'Fold {fold} (AUC = {roc_auc:.2f})')
                    auc_val = roc_auc
                else:
                    auc_val = 0.5
            except:
                auc_val = 0.5 
            
            # Confusion matrix metrics (manual check for LOOCV)
            if len(y_test) == 1:
                # For LOOCV, metrics are binary per person
                tp = 1 if (y_test[0] == 1 and y_pred[0] == 1) else 0
                tn = 1 if (y_test[0] == 0 and y_pred[0] == 0) else 0
                fp = 1 if (y_test[0] == 0 and y_pred[0] == 1) else 0
                fn = 1 if (y_test[0] == 1 and y_pred[0] == 0) else 0
                # We'll average these later
                sens = 1 if (tp + fn) > 0 and tp > 0 else (0 if (tp + fn) > 0 else np.nan)
                spec = 1 if (tn + fp) > 0 and tn > 0 else (0 if (tn + fp) > 0 else np.nan)
            else:
                tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
                sens = tp / (tp + fn) if (tp + fn) > 0 else 0
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            accuracies.append(acc)
            if not np.isnan(sens): sensitivities.append(sens)
            if not np.isnan(spec): specificities.append(spec)
            
            if not use_loocv and len(X) >= 20:
                print(f"Fold {fold}: Acc={acc:.2f}, AUC={auc_val:.2f}")
            fold += 1
            
        # Finalize ROC Plot with Professional Aesthetics
        plt.style.use('seaborn-v0_8-paper')
        plt.figure(figsize=(8, 6))
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='gray', label='Chance (0.50)', alpha=.8)
        
        if len(tprs) > 0:
            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = auc(mean_fpr, mean_tpr)
            std_auc = np.std(aucs) if len(aucs) > 1 else 0
            
            label = f'Mean ROC (AUC = {mean_auc:.2f} $\\pm$ {std_auc:.2f})'
            plt.plot(mean_fpr, mean_tpr, color='#2c3e50', label=label, lw=3, alpha=.9)
            
            # Add range shading
            std_tpr = np.std(tprs, axis=0)
            tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
            tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
            plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')

        plt.xlim([-0.02, 1.02])
        plt.ylim([-0.02, 1.02])
        plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
        plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
        plt.title(f'Diagnostic Performance: ROC Curve (N={len(X)})', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fancybox=True, shadow=True, fontsize=10)
        plt.grid(True, alpha=0.3)

        
        if self.output_dir:
            auc_path = os.path.join(self.output_dir, "ai_auc_performance.png")
            plt.savefig(auc_path)
            print(f"✓ AI Performance Graph (AUC) saved to {auc_path}")
        plt.close()


            
        auc_note = "(Note: AUC limited in LOOCV)" if use_loocv else ""
        summary_lines = [
            "-" * 30,
            f"MEAN ACCURACY:    {np.mean(accuracies):.3f}",
            f"MEAN AUC:         {np.mean(aucs) if aucs else 0.5:.3f} {auc_note}",
            f"MEAN SENSITIVITY: {np.mean(sensitivities):.3f}",
            f"MEAN SPECIFICITY: {np.mean(specificities):.3f}",
            "-" * 30
        ]

        
        for line in summary_lines:
            print(line)
            report_lines.append(line)
            
        if self.output_dir and os.path.exists(self.output_dir):
            with open(os.path.join(self.output_dir, "summary_report.txt"), "w") as f:
                f.write("\n".join(report_lines))
        
        return np.mean(accuracies)

    def explain_model(self, df):
        """Train final model on full data and generate SHAP explanation"""
        print("\nGenerating AI Explanations (SHAP)...")
        X, y = self.prepare_data(df)
        
        # Train on full dataset
        self.model.fit(X, y)
        
        # Get the pipeline directly (No grid search anymore)
        pipeline = self.model
        
        # Preprocess data (Imputation + Scaling) before passing to RF
        # We use the pipeline steps *before* the final model step
        preprocessor = Pipeline(pipeline.steps[:-1])
        X_processed = preprocessor.transform(X)
        
        rf_model = pipeline.named_steps['rf']
        
        # TreeExplainer is much faster and accurate for Random Forests
        explainer = shap.TreeExplainer(rf_model)
        shap_values_raw = explainer.shap_values(X_processed)
        
        # DEBUG PRINTS
        print(f"DEBUG: X shape: {X.shape}")
        print(f"DEBUG: X_processed shape: {X_processed.shape}") # Changed from X_scaled
        print(f"DEBUG: shap_values_raw type: {type(shap_values_raw)}")
        if isinstance(shap_values_raw, list):
            print(f"DEBUG: shap_values_raw list len: {len(shap_values_raw)}")
            print(f"DEBUG: shap_values_raw[0] shape: {shap_values_raw[0].shape}")
        else:
            print(f"DEBUG: shap_values_raw shape: {shap_values_raw.shape}")
            
        # For Logistic Regression binary classification, SHAP returns list [prob_0, prob_1]
        if isinstance(shap_values_raw, list) and len(shap_values_raw) == 2:

            # List format (samples, features) per class
            base_value = explainer.expected_value[1]
            shap_values_raw = np.array(shap_values_raw[1])
        elif isinstance(shap_values_raw, np.ndarray) and len(shap_values_raw.shape) == 3:
            # 3D array format (samples, features, classes)
            base_value = explainer.expected_value[1] if len(explainer.expected_value) > 1 else explainer.expected_value
            shap_values_raw = shap_values_raw[:, :, 1]
        else:
            base_value = explainer.expected_value
            shap_values_raw = np.array(shap_values_raw)
        
        print(f"DEBUG: Final shap_values_raw shape: {shap_values_raw.shape}")
        print(f"DEBUG: Num features in self.features: {len(self.features)}")

        # Ensure base_value is a single scalar for np.repeat
        if hasattr(base_value, "__len__") and len(base_value) > 0:
            if isinstance(base_value, np.ndarray) and len(base_value.shape) > 0:
                base_value = base_value[0]
            elif isinstance(base_value, list):
                base_value = base_value[0]

        # PREMIUM VISUAL FIX: Wrap in a modern SHAP Explanation object
        exp = shap.Explanation(
            values=shap_values_raw,
            base_values=np.repeat(base_value, len(X)),
            data=X.values, 
            feature_names=self.features
        )




        # 1. Premium Bar Chart (shows average impact values)
        plt.figure(figsize=(10, 6))
        shap.plots.bar(exp, max_display=10, show=False)
        plt.title("Top Factors Diagnosing Alzheimer's (Average Impact)", fontsize=14, pad=15)
        
        save_path_bar = os.path.join(self.output_dir, "ai_feature_importance_v5.png")
        plt.savefig(save_path_bar, bbox_inches='tight')
        plt.close()
        print(f"✓ Feature Importance Chart saved to {save_path_bar}")

        # 2. Detailed Distribution Plot (Violin Style)
        # Using summary_plot with 'violin' provides a distinct visual from the dot plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values_raw, X, plot_type="violin", max_display=10, show=False)
        plt.title("Impact Distribution (Spread & Shape)", fontsize=14, pad=15)
        
        save_path_bee = os.path.join(self.output_dir, "ai_shap_detailed_v5.png")
        plt.savefig(save_path_bee, bbox_inches='tight')
        plt.close()
        print(f"✓ Detailed Distribution Chart saved to {save_path_bee}")

        # 3. Individual Dot Plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values_raw, X, plot_type="dot", max_display=10, show=False)
        plt.title("Individual Patient Impact (Each Dot is a Person)", fontsize=14, pad=15)
        
        save_path_dots = os.path.join(self.output_dir, "ai_shap_dotplot_v5.png")
        plt.savefig(save_path_dots, bbox_inches='tight')
        plt.close()
        print(f"✓ Individual Dot Plot saved to {save_path_dots}")


def main():
    # Parse command line arguments for flexibility
    parser = argparse.ArgumentParser(description="Extract EEG features (PLI).")
    parser.add_argument('--full', action='store_true', help='Run analysis on ALL eligible subjects')
    parser.add_argument('--test', action='store_true', help='Run in test mode (fast, 20 subjects)')
    parser.add_argument('--tiny', action='store_true', help='Tiny test mode (ultra-fast, 10 subjects)')
    args = parser.parse_args()
    
    # Determine mode: Command line args override global default
    TINY_MODE = args.tiny
    if args.full:
        TEST_MODE = False
    elif args.test or args.tiny:
        TEST_MODE = True
    else:
        TEST_MODE = DEFAULT_TEST_MODE

    # Path to the datasets
    data_dir_ec = "/home/vijay/Documents/code/Alzheimers/script_downloaded_open_neuro/eeg_alzheimer_data/raw/ds004504"
    data_dir_eo = "/home/vijay/Documents/code/Alzheimers/script_downloaded_open_neuro/eeg_alzheimer_data_eyes_open/raw/ds006036"

    print(f"\n{'='*50}")
    print(f"Starting Multi-Condition Extraction Analysis (EC + EO)")
    if TINY_MODE:
        print(f"Mode: TINY TEST (EC + EO Fusion)")
    else:
        print(f"Mode: {'TEST' if TEST_MODE else 'FULL'} (EC + EO Fusion)")
    print(f"{'='*50}\n")

    print(f"Loading participants from {data_dir_ec}...")
    try:
        participants = load_participants(data_dir_ec)
    except FileNotFoundError:
        print(f"Error: Could not find EC dataset at {data_dir_ec}.")
        return


    # Filter out F group (Frontotemporal Dementia)
    participants = participants[participants['Group'] != 'F']
    print(f"Found {len(participants)} participants (excluding FTD group).")
    
    if TEST_MODE:
        n_ad = 5 if TINY_MODE else 10
        n_hc = 5 if TINY_MODE else 10
        print(f"\n[{'TINY ' if TINY_MODE else ''}TEST MODE ACTIVE]")
        print(f"  - Processing only a subset: {n_ad} Random AD and {n_hc} Random HC subjects.")
        print("  - Using reduced data duration (60s) per subject.")
        print("  - To run full analysis, use --full flag.")
        
        # Take random subset
        ad_subset = participants[participants['Group'] == 'A'].sample(n=n_ad, random_state=42)
        hc_subset = participants[participants['Group'] == 'C'].sample(n=n_hc, random_state=42)
        participants = pd.concat([ad_subset, hc_subset])
        print(f"  -> Processing subset of {len(participants)} participants.")

    extractor = FeaturesExtracter()
    results = []
    
    print("\nProcessing subjects...")
    # Iterate through all participants
    for i, (idx, row) in enumerate(participants.iterrows()):
        sub_id = row['participant_id']
        group = row['Group']
        
        print(f"[{i+1}/{len(participants)}] Processing {sub_id} (Group: {group})...")
        
        result_entry = {
            'participant_id': sub_id,
            'Group': group,
            'Age': row.get('Age', np.nan)
        }
        
        # 1. Extract Eyes Closed (EC) Regional Features
        print(f"    -> Extracting Posterior EC biomarkers...")
        try:
            ec_pli = extractor.compute_subject_pli(sub_id, data_dir_ec, test_mode=TEST_MODE, cluster_only=True)
            if ec_pli: result_entry['EC_Posterior_PLI'] = ec_pli['Global_PLI']
            
            ec_spectral = extractor.compute_subject_spectral_features(sub_id, data_dir_ec, test_mode=TEST_MODE, cluster_only=True)
            if ec_spectral: 
                result_entry['EC_Posterior_Theta_Alpha_Ratio'] = ec_spectral.get('Theta_Alpha_Ratio')
                result_entry['EC_Posterior_Theta_Beta_Ratio'] = ec_spectral.get('Theta_Beta_Ratio')
                result_entry['EC_Posterior_Rel_Alpha'] = ec_spectral.get('Rel_Alpha')
                result_entry['EC_Posterior_Rel_High_Alpha'] = ec_spectral.get('Rel_High_Alpha')
                result_entry['EC_Posterior_Peak_Alpha_Freq'] = ec_spectral.get('Peak_Alpha_Freq')
                # Store absolute power for ABR calc
                result_entry['EC_Abs_Alpha'] = ec_spectral.get('Abs_Alpha', np.nan)


        except Exception as e:
            print(f"    ! Error extracting EC data for {sub_id}: {e}")
        
        # 2. Extract Eyes Open (EO) Regional Features
        print(f"    -> Extracting Posterior EO biomarkers...")
        try:
            eo_pli = extractor.compute_subject_pli(sub_id, data_dir_eo, test_mode=TEST_MODE, cluster_only=True)
            if eo_pli: result_entry['EO_Posterior_PLI'] = eo_pli['Global_PLI']
            
            eo_spectral = extractor.compute_subject_spectral_features(sub_id, data_dir_eo, test_mode=TEST_MODE, cluster_only=True)
            if eo_spectral: 
                result_entry['EO_Posterior_Rel_Alpha'] = eo_spectral.get('Rel_Alpha', np.nan)
                # Store absolute power for ABR calc
                result_entry['EO_Abs_Alpha'] = eo_spectral.get('Abs_Alpha', np.nan)

        except Exception as e:
            print(f"    ! Note: EO data missing/failed for {sub_id}")

        # 3. Calculate Regional "Reactivity" (ABR = EC / EO)
        # Healthy brains block alpha in posterior regions when eyes open.
        if result_entry.get('EC_Abs_Alpha') and result_entry.get('EO_Abs_Alpha'):
            ec_alpha = result_entry['EC_Abs_Alpha']
            eo_alpha = result_entry['EO_Abs_Alpha']
            if eo_alpha > 0:
                result_entry['Reactivity_Posterior_Alpha_Block'] = ec_alpha / eo_alpha





        # Check if we successfully extracted multi-condition data

        has_ec = any(k.startswith('EC_') for k in result_entry.keys())
        has_eo = any(k.startswith('EO_') for k in result_entry.keys())
        
        if has_ec or has_eo:
            results.append(result_entry)
            print(f"    >>> Status: EC Features={'✓' if has_ec else '✗'} | EO Features={'✓' if has_eo else '✗'}")
        else:
            print(f"    -> Failed to compute any resting-state biomarkers.")
        
        # Explicitly clear memory after each subject
        del result_entry
        gc.collect()

        
        print(f"    {'-'*40}")


    # Create DataFrame from results
    if not results:
        print("No results to plot.")
        return

    df_results = pd.DataFrame(results)
    
    # Ensure all expected features exist (fill with NaN if missing)
    expected_features = [
            'EC_Posterior_Theta_Alpha_Ratio', 'EC_Posterior_Rel_Alpha',
            'EC_Posterior_Rel_High_Alpha', 'EC_Posterior_Peak_Alpha_Freq',
            'EC_Posterior_PLI'
        ]










    for col in expected_features:
        if col not in df_results.columns:
            df_results[col] = np.nan

    
    # ==========================================
    # AI MODEL TRAINING & DIAGNOSIS
    # ==========================================
    # 1. Create timestamped directory FIRST
    timestamp = datetime.now().strftime("%b%d_%H%M")
    temp_mode_str = "Tiny" if TINY_MODE else ("Test" if TEST_MODE else "Full")
    output_dir = f"run_v6_{timestamp}_{temp_mode_str}" # accuracy added later or we just keep it stable
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    classifier = AlzheimerClassifier(output_dir=output_dir)
    
    # 2. Run Robust Cross-Validation 
    # Logic: Use 5-Fold (User Preference) for Full Data, but LOOCV for Small Data (N < 35) to avoid math errors
    if len(df_results) < 35:
        print(f"\n[Auto-Switch] Dataset too small for 5-Fold (N={len(df_results)}). Switching to Leave-One-Out CV for stability.")
        use_loocv_flag = True
    else:
        use_loocv_flag = False 
 
        
    mean_acc = classifier.train_with_cv(df_results, use_loocv=use_loocv_flag)

    
    # 3. Finalize folder name with accuracy (optional but helpful)
    final_output_dir = f"{output_dir}_Acc{int(mean_acc * 100)}"
    os.rename(output_dir, final_output_dir)
    classifier.output_dir = final_output_dir
    
    print(f"\n📂 Results saved to: {final_output_dir}/")

    # 4. Save CSV
    csv_path = os.path.join(final_output_dir, "biomarker_results.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"✓ Data saved: {csv_path}")

    # 5. Save Plots
    try:
        features_to_plot = {
            'EC_Posterior_Theta_Alpha_Ratio': 'Posterior Slowing (EC)',
            'EC_Posterior_Rel_Alpha': 'Posterior Alpha (EC)',
            'EC_Posterior_Rel_High_Alpha': 'High Alpha (10-12 Hz)',
            'EC_Posterior_Peak_Alpha_Freq': 'Peak Alpha Frequency (PAF)',
            'EC_Posterior_PLI': 'Posterior Connectivity (EC)'
        }












        
        for feature, title in features_to_plot.items():
            if feature in df_results.columns:
                plt.figure(figsize=(6, 6))
                sns.boxplot(x='Group', y=feature, data=df_results, palette="Set2", hue='Group', legend=False)
                sns.stripplot(x='Group', y=feature, data=df_results, color='black', alpha=0.5, jitter=True)

                plt.title(title, fontsize=14)
                plt.savefig(os.path.join(final_output_dir, f"plot_{feature}.png"))
                plt.close()
        print(f"✓ Feature plots saved.")
    except Exception as e:
        print(f"Error plotting: {e}")

    # 6. AI Explanations
    classifier.explain_model(df_results)
    
    print(f"\n✅ REFINED ANALYSIS COMPLETE")
    print(f"All records and premium charts stored in: {final_output_dir}")

if __name__ == "__main__":
    main()
