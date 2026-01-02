import argparse
import pandas as pd
import mne
import numpy as np
import warnings
import os
from datetime import datetime
from scipy import signal
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, GridSearchCV, LeaveOneOut
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from scipy.stats import mannwhitneyu

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
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

    def compute_subject_spectral_features(self, subject_id, data_dir, test_mode=False):
        """
        Compute relative spectral power for all defined bands.
        Returns a dictionary of features: { 'Rel_Power_Delta': val, ... }
        """
        try:
            raw = load_subject_eeg(subject_id, data_dir)
            
            # Standardize Temporal Sampling: Use same 30s window (10-40s) for all features
            if raw.times[-1] > 40:
                raw.crop(tmin=10, tmax=40)
            elif raw.times[-1] > 30:
                raw.crop(tmin=0, tmax=30)
            
            
            raw.pick(['eeg'], exclude='bads')
            
            # Broadband filter for spectral analysis
            raw.filter(0.5, 45, verbose=False)
            
            # Epochs
            epochs = mne.make_fixed_length_epochs(raw, duration=5.0, preload=True, verbose=False)
            data = epochs.get_data() # (n_epochs, n_channels, n_times)
            sfreq = 500
         
            # Compute PSD using Welch's method
            # nperseg = min(500, data.shape[2]) # 1 sec window
            freqs, psd = signal.welch(data, fs=sfreq, nperseg=500, axis=-1)
            # psd shape: (n_epochs, n_channels, n_freqs)
            
            # Average PSD across epochs and channels -> Global PSD
            # shape: (n_freqs,)
            avg_psd = np.mean(psd, axis=(0, 1))
            
            features = {}
            total_power = 0
            
            # Calculate power for each band
            for band, (fmin, fmax) in self.BANDS.items():
                idx_band = np.logical_and(freqs >= fmin, freqs <= fmax)
                power = trapezoid(avg_psd[idx_band], freqs[idx_band])
                features[f'Abs_{band}'] = power
                total_power += power
            
            # Calculate Relative Power
            for band in self.BANDS:
                features[f'Rel_{band}'] = features[f'Abs_{band}'] / total_power if total_power > 0 else 0
            
            # Calculate spectral band ratios
            rel_theta = features.get('Rel_Theta', 0)
            rel_beta = features.get('Rel_Beta', 0)
            rel_alpha = features.get('Rel_Alpha', 0)

            features['Theta_Beta_Ratio'] = rel_theta / rel_beta if rel_beta > 0 else 0
            features['Alpha_Theta_Ratio'] = rel_alpha / rel_theta if rel_theta > 0 else 0
            features['Theta_Alpha_Ratio'] = rel_theta / rel_alpha if rel_alpha > 0 else 0  # DAR (gold standard)
            
            return features
            
        except Exception as e:
            print(f"Error extracting spectral features for {subject_id}: {e}")
            return None

    def compute_subject_pli(self, subject_id, data_dir, test_mode=False):
        """
        Load data and calculate the global PLI for a single subject.
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

    @staticmethod
    def _higuchi_fd(x, kmax=100):
        """Calculate Higuchi Fractal Dimension (Complexity)."""
        L = []
        x = np.array(x, dtype=float)
        N = len(x)
        for k in range(1, kmax + 1):
            Lk = []
            for m in range(k):
                Lmk = 0
                for i in range(1, int(np.floor((N - m) / k))):
                    Lmk += abs(x[m + i * k] - x[m + (i - 1) * k])
                Lmk = Lmk * (N - 1) / np.floor((N - m) / float(k)) / k
                Lk.append(Lmk)
            L.append(np.log(np.mean(Lk)))
        
        # Fit line to log(L) vs log(1/k) to find slope (dimension) using simple least squares
        k_idxs = np.arange(1, kmax + 1)
        # Handle division by zero/log issues gracefully
        valid = k_idxs > 0
        x_reg = np.log(1.0 / k_idxs[valid])
        y_reg = L
        
        # Slope is the HFD
        slope, _ = np.polyfit(x_reg, y_reg, 1)
        return slope

    @staticmethod
    def _sample_entropy(x, m=2, r=0.2):
        """Optimized Sample Entropy using vectorization and downsampling."""
        # Standardize: Downsample to ~100Hz if signal is large (e.g. 500Hz -> 100Hz)
        # This keeps the window size manageable (3000 points) while preserving EEG complexity
        if len(x) > 5000:
            x = x[::5] 
            
        N = len(x)
        # Normalize for pattern matching stability
        x = (x - np.mean(x)) / (np.std(x) + 1e-12)
        
        def _get_matches(m_val):
            # Create template matrix: (N-m, m)
            patterns = np.zeros((N - m_val, m_val))
            for k in range(m_val):
                patterns[:, k] = x[k : N - m_val + k]
            
            count = 0
            # Vectorized comparison for each template against all subsequent templates
            for i in range(len(patterns) - 1):
                # Chebyshev distance: max absolute difference
                dist = np.max(np.abs(patterns[i+1:] - patterns[i]), axis=1)
                count += np.sum(dist < r)
            return count

        # A: matches of length m+1, B: matches of length m
        A = _get_matches(m + 1)
        B = _get_matches(m)
        
        if A == 0 or B == 0: return 0
        return -np.log(A / B)

    
    def compute_peak_alpha_frequency(self, subject_id, data_dir, test_mode=False):
        """Extract Peak Alpha Frequency (PAF) - Slowing biomarker."""
        raw = load_subject_eeg(subject_id, data_dir)
        if raw is None: return None
        
        # Standardize Temporal Sampling: Use same 30s window (10-40s) for all features
        if raw.times[-1] > 40:
            raw.crop(tmin=10, tmax=40)
        elif raw.times[-1] > 30:
            raw.crop(tmin=0, tmax=30)
        
        
        try:
            # Filter for Alpha band
            raw.filter(8, 12, verbose=False)
            
            # Get data (average of all channels)
            data = raw.get_data().mean(axis=0)
            sfreq = int(raw.info['sfreq'])
            
            # Compute PSD
            freqs, psd = signal.welch(data, sfreq, nperseg=sfreq*2)
            
            # Find peak in 8-12 Hz
            alpha_mask = (freqs >= 8) & (freqs <= 12)
            alpha_freqs = freqs[alpha_mask]
            alpha_psd = psd[alpha_mask]
            
            if len(alpha_psd) > 0:
                peak_idx = np.argmax(alpha_psd)
                paf = alpha_freqs[peak_idx]
                return {'Peak_Alpha_Freq': paf}
            else:
                return None
                
        except Exception as e:
            print(f"    ! Error computing PAF: {e}")
            return None
    
    def compute_subject_complexity(self, subject_id, data_dir, test_mode=False):
        """Extract Sample Entropy (SampEn) - Gold Standard Complexity."""
        raw = load_subject_eeg(subject_id, data_dir)
        if raw is None: return None
            
        # Standardize Temporal Sampling: Use same 30s window (10-40s)
        if raw.times[-1] > 40:
            raw.crop(tmin=10, tmax=40)
        elif raw.times[-1] > 30:
            raw.crop(tmin=0, tmax=30)
        
        try:
            # Filter for wide band
            raw.filter(1, 45, verbose=False)
            
            # Get data (average of all channels)
            data = raw.get_data().mean(axis=0)
            sfreq = int(raw.info['sfreq'])
            
            # Sample Entropy
            sampen = self._sample_entropy(data, m=2, r=0.2)
            
            return {'Sample_Entropy': sampen}
                
        except Exception as e:
            print(f"    ! Error computing SampEn: {e}")
            return None


class AlzheimerClassifier:
    """
    AI Model to classify Alzheimer's vs Control using EEG features and clinical data.
    Uses SVM with GridSearchCV for hyperparameter optimization and SHAP for explainability.
    """
    def __init__(self, output_dir="model_results"):
        self.output_dir = output_dir
        
        # Expert advice: Switch to simpler Logistic Regression for small n=65
        lr = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000, penalty='l2')
        
        # Create a pipeline: Imputation -> Scaling (Normalization) -> Classifier
        # Normalization is critical for Logistic Regression L2 penalty
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('lr', lr)
        ])
        
        # Simple parameter grid to prevent overfitting
        param_grid = {
            'lr__C': [0.1, 1, 10]
        }
        self.model = GridSearchCV(pipeline, param_grid, cv=3, scoring='roc_auc')
        
        # EXPERT FUSION: Utilizing both Eyes Closed (EC) and Eyes Open (EO) biomarkers
        self.features = [
            'Age',
            'EC_Global_PLI',
            'EO_Global_PLI',
            'EC_Theta_Alpha_Ratio',
            'EO_Theta_Alpha_Ratio',
            'EC_Sample_Entropy',
            'EO_Sample_Entropy'
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
        if use_loocv or len(X) < 20:
            print("\nStarting Leave-One-Out Cross-Validation (LOOCV)...")
            cv = LeaveOneOut()
        else:
            print("\nStarting Stratified 5-Fold Cross-Validation...")
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
        accuracies, aucs, sensitivities, specificities = [], [], [], []
        fold = 1
        
        report_lines = ["Starting Cross-Validation Summary", "="*30]
        report_lines.extend(stats_report)
        report_lines.append(f"\nData Prepared: {len(X)} samples. (A: {sum(y==1)}, C: {sum(y==0)})")
        
        for train_index, test_index in cv.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            try:
                # Use predict_proba for Logistic Regression AUC
                y_prob = self.model.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.5
            except:
                auc = 0.5 
            
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
            if auc != 0.5 or len(y_test) > 1: aucs.append(auc)
            
            if not use_loocv and len(X) >= 20:
                print(f"Fold {fold}: Acc={acc:.2f}, AUC={auc:.2f}")
            fold += 1
            
        summary_lines = [
            "-" * 30,
            f"MEAN ACCURACY:    {np.mean(accuracies):.3f}",
            f"MEAN AUC:         {np.mean(aucs) if aucs else 0.5:.3f} (Note: AUC limited in LOOCV)",
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
        
        # Get the best estimator for SHAP
        best_pipeline = self.model.best_estimator_
        X_scaled = best_pipeline.named_steps['scaler'].transform(X)
        lr_model = best_pipeline.named_steps['lr']
        
        # BACKGROUND: use predict_proba for SHAP with Logistic Regression
        background = shap.kmeans(X_scaled, 5) 
        explainer = shap.KernelExplainer(lr_model.predict_proba, background)
        shap_values_raw = explainer.shap_values(X_scaled)
        
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
        
        # 1. Extract Eyes Closed (EC) Features
        print(f"    -> Extracting Eyes Closed (EC) biomarkers...")
        try:
            ec_pli = extractor.compute_subject_pli(sub_id, data_dir_ec, test_mode=TEST_MODE)
            if ec_pli: result_entry['EC_Global_PLI'] = ec_pli['Global_PLI']
            
            ec_spectral = extractor.compute_subject_spectral_features(sub_id, data_dir_ec, test_mode=TEST_MODE)
            if ec_spectral: result_entry['EC_Theta_Alpha_Ratio'] = ec_spectral['Theta_Alpha_Ratio']
            
            ec_complexity = extractor.compute_subject_complexity(sub_id, data_dir_ec, test_mode=TEST_MODE)
            if ec_complexity: result_entry['EC_Sample_Entropy'] = ec_complexity['Sample_Entropy']
        except Exception as e:
            print(f"    ! Error extracting EC data for {sub_id}: {e}")
        
        # 2. Extract Eyes Open (EO) Features
        print(f"    -> Extracting Eyes Open (EO) biomarkers...")
        try:
            eo_pli = extractor.compute_subject_pli(sub_id, data_dir_eo, test_mode=TEST_MODE)
            if eo_pli: result_entry['EO_Global_PLI'] = eo_pli['Global_PLI']
            
            eo_spectral = extractor.compute_subject_spectral_features(sub_id, data_dir_eo, test_mode=TEST_MODE)
            if eo_spectral: result_entry['EO_Theta_Alpha_Ratio'] = eo_spectral['Theta_Alpha_Ratio']
            
            eo_complexity = extractor.compute_subject_complexity(sub_id, data_dir_eo, test_mode=TEST_MODE)
            if eo_complexity: result_entry['EO_Sample_Entropy'] = eo_complexity['Sample_Entropy']
        except Exception as e:
            print(f"    ! Note: EO data missing/failed for {sub_id}")

        # Check if we successfully extracted multi-condition data
        has_ec = any(k.startswith('EC_') for k in result_entry.keys())
        has_eo = any(k.startswith('EO_') for k in result_entry.keys())
        
        if has_ec or has_eo:
            results.append(result_entry)
            print(f"    >>> Status: EC Features={'✓' if has_ec else '✗'} | EO Features={'✓' if has_eo else '✗'}")
        else:
            print(f"    -> Failed to compute any resting-state biomarkers.")
        
        print(f"    {'-'*40}")


    # Create DataFrame from results
    if not results:
        print("No results to plot.")
        return

    df_results = pd.DataFrame(results)
    
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
    # Use LOOCV for full analysis to maximize every sample
    is_full_analysis = not TEST_MODE
    mean_acc = classifier.train_with_cv(df_results, use_loocv=is_full_analysis)
    
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
            'EC_Global_PLI': 'Functional Connectivity (EC PLI)',
            'EO_Global_PLI': 'Functional Connectivity (EO PLI)',
            'EC_Theta_Alpha_Ratio': 'Theta/Alpha Ratio (EC DAR)',
            'EO_Theta_Alpha_Ratio': 'Theta/Alpha Ratio (EO DAR)',
            'EC_Sample_Entropy': 'Signal Complexity (EC SampEn)',
            'EO_Sample_Entropy': 'Signal Complexity (EO SampEn)',
            'Age': 'Patient Age'
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
