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

matplotlib.use('Agg')
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
import sys

# Force unbuffered output
sys.stdout = sys.stderr = open(sys.stdout.fileno(), 'w', buffering=1)

warnings.filterwarnings("ignore", category=RuntimeWarning, module="mne")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# ==========================================
# CONFIGURATION
# ==========================================
DEFAULT_TEST_MODE = True


# ==========================================


def load_participants(data_dir):
    """Load participants.tsv file."""
    participants_path = os.path.join(data_dir, 'participants.tsv')
    if not os.path.exists(participants_path):
        raise FileNotFoundError(f"participants.tsv not found at {participants_path}")

    return pd.read_csv(participants_path, sep='\t')


def load_subject_eeg(subject_id, data_dir):
    """
    Load EEG data for a specific subject.
    Prefer preprocessed data in 'derivatives' folder if available.
    """
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
    BANDS = {
        'Delta': (0.5, 4),
        'Theta': (4, 8),
        'Alpha': (8, 12),
        'Beta': (12, 30),
        'Gamma': (30, 45)
    }

    POSTERIOR_TEMPORAL = ['P3', 'P4', 'T5', 'T6', 'O1', 'O2']
    POSTERIOR_LEFT = ['P3', 'T5', 'O1']
    POSTERIOR_RIGHT = ['P4', 'T6', 'O2']

    def compute_subject_spectral_features(self, subject_id, data_dir, test_mode=False, cluster_only=True):
        """
        Compute relative spectral power for the AD-critical Posterior/Temporal cluster.
        """
        try:
            raw = load_subject_eeg(subject_id, data_dir)

            if raw.times[-1] > 40:
                raw.crop(tmin=10, tmax=40)
            elif raw.times[-1] > 30:
                raw.crop(tmin=0, tmax=30)

            raw.pick(['eeg'], exclude='bads')

            try:
                montage = mne.channels.make_standard_montage('standard_1020')
                raw.set_montage(montage, on_missing='ignore', verbose=False)
            except:
                pass

            try:
                raw.set_eeg_reference('average', projection=False, verbose=False)
            except:
                pass

            raw.filter(0.5, 45, verbose=False)
            epochs = mne.make_fixed_length_epochs(raw, duration=5.0, preload=True, verbose=False)

            try:
                epochs.drop_bad(reject=dict(eeg=100e-6), verbose=False)
            except:
                pass

            if len(epochs) == 0:
                return None

            ch_names = epochs.ch_names

            if cluster_only:
                cluster_indices = [i for i, ch in enumerate(ch_names) if ch.upper() in self.POSTERIOR_TEMPORAL]
                if not cluster_indices:
                    print(f"    ! Warning: Cluster electrodes {self.POSTERIOR_TEMPORAL} not found in {subject_id}. Using all sensors.")
                    filtered_data = epochs.get_data()
                    subset_ch_names = ch_names
                else:
                    filtered_data = epochs.get_data(picks=cluster_indices)
                    subset_ch_names = [ch_names[i] for i in cluster_indices]
            else:
                filtered_data = epochs.get_data()
                subset_ch_names = ch_names

            sfreq = 500

            freqs, psd = signal.welch(filtered_data, fs=sfreq, nperseg=500, axis=-1)

            avg_psd = np.mean(psd, axis=(0, 1))
            results = {}
            total_psd = trapezoid(avg_psd, freqs)

            delta_val = trapezoid(avg_psd[(freqs >= 1) & (freqs <= 4)], freqs[(freqs >= 1) & (freqs <= 4)])
            theta_val = trapezoid(avg_psd[(freqs >= 4) & (freqs <= 8)], freqs[(freqs >= 4) & (freqs <= 8)])
            alpha_val = trapezoid(avg_psd[(freqs >= 8) & (freqs <= 12)], freqs[(freqs >= 8) & (freqs <= 12)])
            beta_val = trapezoid(avg_psd[(freqs >= 12) & (freqs <= 30)], freqs[(freqs >= 12) & (freqs <= 30)])
            gamma_val = trapezoid(avg_psd[(freqs >= 30) & (freqs <= 45)], freqs[(freqs >= 30) & (freqs <= 45)])

            if alpha_val > 0:
                results['Theta_Alpha_Ratio'] = theta_val / alpha_val
            if beta_val > 0:
                results['Theta_Beta_Ratio'] = theta_val / beta_val

            if total_psd > 0:
                results['Rel_Alpha'] = alpha_val / total_psd
                results['Rel_Theta'] = theta_val / total_psd
                results['Rel_Gamma'] = gamma_val / total_psd

            high_alpha = trapezoid(avg_psd[(freqs >= 10) & (freqs <= 12)], freqs[(freqs >= 10) & (freqs <= 12)])
            if total_psd > 0:
                results['Rel_High_Alpha'] = high_alpha / total_psd

            # Peak Alpha Frequency (PAF)
            alpha_mask = (freqs >= 8) & (freqs <= 12)
            if np.any(alpha_mask):
                results['Peak_Alpha_Freq'] = freqs[alpha_mask][np.argmax(avg_psd[alpha_mask])]

            # DAR INSTABILITY
            epoch_avg_psd = np.mean(psd, axis=1)
            epoch_theta = trapezoid(epoch_avg_psd[:, (freqs >= 4) & (freqs <= 8)], freqs[(freqs >= 4) & (freqs <= 8)], axis=-1)
            epoch_alpha = trapezoid(epoch_avg_psd[:, (freqs >= 8) & (freqs <= 12)], freqs[(freqs >= 8) & (freqs <= 12)], axis=-1)

            valid_epochs = epoch_alpha > 0
            if np.any(valid_epochs):
                dar_per_epoch = epoch_theta[valid_epochs] / epoch_alpha[valid_epochs]
                results['DAR_Instability'] = np.std(dar_per_epoch)
            else:
                results['DAR_Instability'] = 0

            # HEMISPHERIC ASYMMETRY
            left_indices = [i for i, ch in enumerate(subset_ch_names) if ch.upper() in self.POSTERIOR_LEFT]
            right_indices = [i for i, ch in enumerate(subset_ch_names) if ch.upper() in self.POSTERIOR_RIGHT]

            if left_indices and right_indices:
                left_psd = np.mean(psd[:, left_indices, :], axis=(0, 1))
                right_psd = np.mean(psd[:, right_indices, :], axis=(0, 1))

                left_alpha = trapezoid(left_psd[(freqs >= 8) & (freqs <= 12)], freqs[(freqs >= 8) & (freqs <= 12)])
                right_alpha = trapezoid(right_psd[(freqs >= 8) & (freqs <= 12)], freqs[(freqs >= 8) & (freqs <= 12)])

                if (left_alpha + right_alpha) > 0:
                    results['Alpha_Asymmetry'] = (right_alpha - left_alpha) / (right_alpha + left_alpha)
            else:
                results['Alpha_Asymmetry'] = 0

            results['Abs_Alpha'] = alpha_val
            results['Abs_Theta'] = theta_val

            return results

        except Exception as e:
            print(f"    ! Error computing spectral features: {e}")
            import traceback
            traceback.print_exc()
            return None

    def get_subject_psd(self, subject_id, data_dir):
        """Helper to get raw PSD data for visual comparison."""
        try:
            raw = load_subject_eeg(subject_id, data_dir)
            if raw is None:
                return None, None

            sfreq = raw.info['sfreq']

            if raw.times[-1] > 60:
                raw.crop(tmin=10, tmax=60)

            raw.pick(['eeg'], exclude='bads')
            raw.filter(0.5, 45, verbose=False)
            epochs = mne.make_fixed_length_epochs(raw, duration=5.0, preload=True, verbose=False)

            try:
                epochs.drop_bad(reject=dict(eeg=300e-6), verbose=False)
                if len(epochs) == 0:
                    epochs = mne.make_fixed_length_epochs(raw, duration=5.0, preload=True, verbose=False)
            except:
                pass

            if len(epochs) == 0:
                return None, None

            ch_names = epochs.ch_names
            cluster_indices = [i for i, ch in enumerate(ch_names) if ch.upper() in self.POSTERIOR_TEMPORAL]
            data = epochs.get_data(picks=cluster_indices) if cluster_indices else epochs.get_data()

            freqs, psd = signal.welch(data, fs=sfreq, nperseg=int(sfreq), axis=-1)
            avg_psd = np.mean(psd, axis=(0, 1))
            return freqs, avg_psd
        except:
            return None, None

    def compute_subject_pli(self, subject_id, data_dir, test_mode=False, cluster_only=True):
        """Compute Global Phase Lag Index (Functional Connectivity) in the Alpha band."""
        raw = load_subject_eeg(subject_id, data_dir)
        if raw is None:
            return None

        if raw.times[-1] > 40:
            raw.crop(tmin=10, tmax=40)
        elif raw.times[-1] > 30:
            raw.crop(tmin=0, tmax=30)

        raw.pick(['eeg'], exclude='bads')

        if cluster_only:
            valid_cluster = [ch for ch in raw.ch_names if ch.upper() in self.POSTERIOR_TEMPORAL]
            if len(valid_cluster) >= 2:
                raw.pick(valid_cluster)
            else:
                print(f"    ! Warning: Cluster too small for PLI in {subject_id}. Using all.")

        try:
            try:
                montage = mne.channels.make_standard_montage('standard_1020')
                raw.set_montage(montage, on_missing='ignore', verbose=False)
            except:
                pass

            try:
                raw.set_eeg_reference('average', projection=False, verbose=False)
            except:
                pass

            raw.filter(8, 12, verbose=False)

            epochs = mne.make_fixed_length_epochs(raw, duration=5.0, preload=True, verbose=False)

            try:
                epochs.drop_bad(reject=dict(eeg=150e-6), verbose=False)
            except:
                pass

            if len(epochs) == 0:
                print(f"    -> All epochs rejected (too noisy).")
                return np.nan

            epochs_data = epochs.get_data()
            n_channels = epochs_data.shape[1]

            adj_matrix = np.zeros((n_channels, n_channels))
            count = 0

            for i in range(n_channels):
                for j in range(i + 1, n_channels):
                    sig1 = epochs_data[:, i, :]
                    sig2 = epochs_data[:, j, :]

                    pli = self._compute_pli(sig1, sig2)
                    adj_matrix[i, j] = pli
                    adj_matrix[j, i] = pli
                    count += 1

            if count > 0:
                global_pli = np.mean(adj_matrix[np.triu_indices(n_channels, k=1)])
                return {'Global_PLI': global_pli}
            else:
                return None

        except Exception as e:
            print(f"Skipping {subject_id} due to error: {e}")
            return np.nan

    def _compute_pli(self, signal1, signal2):
        """Compute PLI across all epochs"""
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
        freqs, psd = signal.welch(x, sfreq, nperseg=sfreq * 2)
        psd_norm = psd / np.sum(psd)
        se = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))
        se /= np.log2(len(psd_norm))
        return se

    def compute_subject_complexity(self, subject_id, data_dir, test_mode=False):
        """Deprecated - Focusing on high-accuracy spectral/connectivity features."""
        return None


class AlzheimerClassifier:
    """
    AI Model to classify Alzheimer's vs Control using EEG features and clinical data.
    """

    def __init__(self, output_dir="model_results"):
        self.output_dir = output_dir

        self.model = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('lr', LogisticRegression(
                class_weight='balanced',
                random_state=42,
                max_iter=1000
            ))
        ])

        self.features = [
            'EC_Posterior_Theta_Alpha_Ratio',
            'EC_Posterior_Rel_Alpha',
            'EC_Posterior_Rel_High_Alpha',
            'EC_Posterior_Peak_Alpha_Freq',
            'EC_Posterior_PLI'

        ]


    def prepare_data(self, df):
        """Prepare features (X) and target (y)"""
        df_clean = df[df['Group'].isin(['A', 'C'])].copy()
        df_clean['Target'] = df_clean['Group'].map({'A': 1, 'C': 0})

        X = df_clean[self.features]
        y = df_clean['Target'].values

        nan_counts = X.isna().sum()
        total_rows = len(X)
        if nan_counts.sum() > 0:
            print("\n    ! WARNING: Missing Data Detected:")
            for col, count in nan_counts.items():
                if count > 0:
                    print(f"      - {col}: {count}/{total_rows} missing ({count / total_rows:.1%})")

        print(f"Data Prepared: {len(X)} samples. (A: {sum(y == 1)}, C: {sum(y == 0)})")
        return X, y

    def train_with_cv(self, df, use_loocv=False, compare_rf=False):
        """Train using Cross Validation and print metrics

        Args:
            df: DataFrame with features and labels
            use_loocv: Use Leave-One-Out CV instead of 5-fold
            compare_rf: Also train Random Forest for comparison
        """
        X, y = self.prepare_data(df)

        # Statistical tests (this is fine - descriptive stats, not predictive modeling)
        print("\nStatistical Proof (AD vs HC):")
        stats_report = ["Feature Significance (Mann-Whitney U):", "-" * 40]
        for feature in self.features:
            ad_vals = df[df['Group'] == 'A'][feature].dropna()
            hc_vals = df[df['Group'] == 'C'][feature].dropna()
            stat, p = mannwhitneyu(ad_vals, hc_vals)
            stars = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            line = f"{feature:18} | p={p:.4f} ({stars})"
            print(f"  {line}")
            stats_report.append(line)

        # REMOVED: Correlation plot before CV (data leakage)
        # Will add it to explain_model() function instead

        if use_loocv:
            print("\nStarting Leave-One-Out Cross-Validation (LOOCV)...")
            cv = LeaveOneOut()
        else:
            print("\nStarting Stratified 5-Fold Cross-Validation...")
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Storage for results
        lr_accuracies, lr_aucs, lr_sensitivities, lr_specificities = [], [], [], []
        rf_accuracies, rf_aucs, rf_sensitivities, rf_specificities = [], [], [], []
        lr_tprs, rf_tprs = [], []
        mean_fpr = np.linspace(0, 1, 100)
        fold = 1

        report_lines = ["Starting Cross-Validation Summary", "=" * 30]
        report_lines.extend(stats_report)
        report_lines.append(f"\nData Prepared: {len(X)} samples. (A: {sum(y == 1)}, C: {sum(y == 0)})")

        # Create Random Forest model if comparing
        if compare_rf:
            from sklearn.ensemble import RandomForestClassifier
            rf_model = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
                ('rf', RandomForestClassifier(
                    class_weight='balanced',
                    random_state=42,
                    n_estimators=100,
                    max_depth=5  # Limit depth for small dataset
                ))
            ])

        plt.figure(figsize=(10, 8))

        for train_index, test_index in cv.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # ===== LOGISTIC REGRESSION =====
            self.model.fit(X_train, y_train)
            lr_pred = self.model.predict(X_test)
            lr_acc = accuracy_score(y_test, lr_pred)

            try:
                lr_prob = self.model.predict_proba(X_test)[:, 1]
                if len(np.unique(y_test)) > 1:
                    fpr, tpr, _ = roc_curve(y_test, lr_prob)
                    roc_auc = auc(fpr, tpr)
                    lr_aucs.append(roc_auc)

                    interp_tpr = np.interp(mean_fpr, fpr, tpr)
                    interp_tpr[0] = 0.0
                    lr_tprs.append(interp_tpr)

                    plt.plot(fpr, tpr, lw=1, alpha=0.3, color='blue',
                             label=f'LR Fold {fold} (AUC={roc_auc:.2f})' if fold == 1 else '')
                    lr_auc_val = roc_auc
                else:
                    lr_auc_val = 0.5
            except:
                lr_auc_val = 0.5

            # Calculate sensitivity/specificity for LR
            if len(y_test) == 1:
                tp = 1 if (y_test[0] == 1 and lr_pred[0] == 1) else 0
                tn = 1 if (y_test[0] == 0 and lr_pred[0] == 0) else 0
                fp = 1 if (y_test[0] == 0 and lr_pred[0] == 1) else 0
                fn = 1 if (y_test[0] == 1 and lr_pred[0] == 0) else 0
                lr_sens = 1 if (tp + fn) > 0 and tp > 0 else (0 if (tp + fn) > 0 else np.nan)
                lr_spec = 1 if (tn + fp) > 0 and tn > 0 else (0 if (tn + fp) > 0 else np.nan)
            else:
                tn, fp, fn, tp = confusion_matrix(y_test, lr_pred, labels=[0, 1]).ravel()
                lr_sens = tp / (tp + fn) if (tp + fn) > 0 else 0
                lr_spec = tn / (tn + fp) if (tn + fp) > 0 else 0

            lr_accuracies.append(lr_acc)
            if not np.isnan(lr_sens): lr_sensitivities.append(lr_sens)
            if not np.isnan(lr_spec): lr_specificities.append(lr_spec)

            # ===== RANDOM FOREST (if comparing) =====
            if compare_rf:
                rf_model.fit(X_train, y_train)
                rf_pred = rf_model.predict(X_test)
                rf_acc = accuracy_score(y_test, rf_pred)

                try:
                    rf_prob = rf_model.predict_proba(X_test)[:, 1]
                    if len(np.unique(y_test)) > 1:
                        fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_prob)
                        roc_auc_rf = auc(fpr_rf, tpr_rf)
                        rf_aucs.append(roc_auc_rf)

                        interp_tpr_rf = np.interp(mean_fpr, fpr_rf, tpr_rf)
                        interp_tpr_rf[0] = 0.0
                        rf_tprs.append(interp_tpr_rf)

                        plt.plot(fpr_rf, tpr_rf, lw=1, alpha=0.3, color='red',
                                 label=f'RF Fold {fold} (AUC={roc_auc_rf:.2f})' if fold == 1 else '')
                        rf_auc_val = roc_auc_rf
                    else:
                        rf_auc_val = 0.5
                except:
                    rf_auc_val = 0.5

                # Calculate sensitivity/specificity for RF
                if len(y_test) == 1:
                    tp = 1 if (y_test[0] == 1 and rf_pred[0] == 1) else 0
                    tn = 1 if (y_test[0] == 0 and rf_pred[0] == 0) else 0
                    fp = 1 if (y_test[0] == 0 and rf_pred[0] == 1) else 0
                    fn = 1 if (y_test[0] == 1 and rf_pred[0] == 0) else 0
                    rf_sens = 1 if (tp + fn) > 0 and tp > 0 else (0 if (tp + fn) > 0 else np.nan)
                    rf_spec = 1 if (tn + fp) > 0 and tn > 0 else (0 if (tn + fp) > 0 else np.nan)
                else:
                    tn, fp, fn, tp = confusion_matrix(y_test, rf_pred, labels=[0, 1]).ravel()
                    rf_sens = tp / (tp + fn) if (tp + fn) > 0 else 0
                    rf_spec = tn / (tn + fp) if (tn + fp) > 0 else 0

                rf_accuracies.append(rf_acc)
                if not np.isnan(rf_sens): rf_sensitivities.append(rf_sens)
                if not np.isnan(rf_spec): rf_specificities.append(rf_spec)

                print(
                    f"Fold {fold}: LR Acc={lr_acc:.2f} AUC={lr_auc_val:.2f} | RF Acc={rf_acc:.2f} AUC={rf_auc_val:.2f}")
            else:
                if not use_loocv and len(X) >= 20:
                    print(f"Fold {fold}: Acc={lr_acc:.2f}, AUC={lr_auc_val:.2f}")

            fold += 1

        # Finalize ROC Plot
        plt.style.use('seaborn-v0_8-paper')
        plt.figure(figsize=(8, 6))
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='gray', label='Chance (0.50)', alpha=.8)

        # Plot LR mean curve
        if len(lr_tprs) > 0:
            mean_tpr_lr = np.mean(lr_tprs, axis=0)
            mean_tpr_lr[-1] = 1.0
            mean_auc_lr = auc(mean_fpr, mean_tpr_lr)
            std_auc_lr = np.std(lr_aucs) if len(lr_aucs) > 1 else 0

            label_lr = f'LR Mean ROC (AUC = {mean_auc_lr:.2f} $\\pm$ {std_auc_lr:.2f})'
            plt.plot(mean_fpr, mean_tpr_lr, color='#2c3e50', label=label_lr, lw=3, alpha=.9)

            std_tpr_lr = np.std(lr_tprs, axis=0)
            tprs_upper_lr = np.minimum(mean_tpr_lr + std_tpr_lr, 1)
            tprs_lower_lr = np.maximum(mean_tpr_lr - std_tpr_lr, 0)
            plt.fill_between(mean_fpr, tprs_lower_lr, tprs_upper_lr, color='blue', alpha=.2)

        # Plot RF mean curve if comparing
        if compare_rf and len(rf_tprs) > 0:
            mean_tpr_rf = np.mean(rf_tprs, axis=0)
            mean_tpr_rf[-1] = 1.0
            mean_auc_rf = auc(mean_fpr, mean_tpr_rf)
            std_auc_rf = np.std(rf_aucs) if len(rf_aucs) > 1 else 0

            label_rf = f'RF Mean ROC (AUC = {mean_auc_rf:.2f} $\\pm$ {std_auc_rf:.2f})'
            plt.plot(mean_fpr, mean_tpr_rf, color='#e74c3c', label=label_rf, lw=3, alpha=.9, linestyle='--')

            std_tpr_rf = np.std(rf_tprs, axis=0)
            tprs_upper_rf = np.minimum(mean_tpr_rf + std_tpr_rf, 1)
            tprs_lower_rf = np.maximum(mean_tpr_rf - std_tpr_rf, 0)
            plt.fill_between(mean_fpr, tprs_lower_rf, tprs_upper_rf, color='red', alpha=.2)

        plt.xlim([-0.02, 1.02])
        plt.ylim([-0.02, 1.02])
        plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
        plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)

        title = 'Model Comparison: ROC Curves' if compare_rf else 'Diagnostic Performance: ROC Curve'
        plt.title(f'{title} (N={len(X)})', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fancybox=True, shadow=True, fontsize=10)
        plt.grid(True, alpha=0.3)

        if self.output_dir:
            auc_path = os.path.join(self.output_dir, "ai_auc_performance.png")
            plt.savefig(auc_path)
            print(f"✓ AI Performance Graph (AUC) saved to {auc_path}")
        plt.close()

        # Print results
        auc_note = "(Note: AUC limited in LOOCV)" if use_loocv else ""
        print("\n" + "=" * 60)
        print("LOGISTIC REGRESSION RESULTS:")
        print("-" * 60)
        print(f"MEAN ACCURACY:    {np.mean(lr_accuracies):.3f}")
        print(f"MEAN AUC:         {np.mean(lr_aucs) if lr_aucs else 0.5:.3f} {auc_note}")
        print(f"MEAN SENSITIVITY: {np.mean(lr_sensitivities):.3f}")
        print(f"MEAN SPECIFICITY: {np.mean(lr_specificities):.3f}")

        if compare_rf:
            print("\n" + "=" * 60)
            print("RANDOM FOREST RESULTS:")
            print("-" * 60)
            print(f"MEAN ACCURACY:    {np.mean(rf_accuracies):.3f}")
            print(f"MEAN AUC:         {np.mean(rf_aucs) if rf_aucs else 0.5:.3f} {auc_note}")
            print(f"MEAN SENSITIVITY: {np.mean(rf_sensitivities):.3f}")
            print(f"MEAN SPECIFICITY: {np.mean(rf_specificities):.3f}")

            print("\n" + "=" * 60)
            print("COMPARISON:")
            print("-" * 60)
            acc_diff = np.mean(lr_accuracies) - np.mean(rf_accuracies)
            auc_diff = np.mean(lr_aucs) - np.mean(rf_aucs) if (lr_aucs and rf_aucs) else 0
            print(f"Accuracy Difference (LR - RF): {acc_diff:+.3f}")
            print(f"AUC Difference (LR - RF):      {auc_diff:+.3f}")

            if abs(acc_diff) < 0.02 and abs(auc_diff) < 0.02:
                print("\n⚠️  WARNING: Models perform nearly identically!")
                print("    This suggests features are highly linearly separable.")
                print("    The dataset may have very strong signal-to-noise ratio.")

        print("=" * 60)

        summary_lines = [
            "-" * 30,
            f"MEAN ACCURACY:    {np.mean(lr_accuracies):.3f}",
            f"MEAN AUC:         {np.mean(lr_aucs) if lr_aucs else 0.5:.3f} {auc_note}",
            f"MEAN SENSITIVITY: {np.mean(lr_sensitivities):.3f}",
            f"MEAN SPECIFICITY: {np.mean(lr_specificities):.3f}",
            "-" * 30
        ]

        for line in summary_lines:
            print(line)
            report_lines.append(line)

        # Write comprehensive report including both models
        if self.output_dir and os.path.exists(self.output_dir):
            with open(os.path.join(self.output_dir, "summary_report.txt"), "w") as f:
                # Write base report (stats + LR results)
                f.write("\n".join(report_lines))

                # Add RF results if comparing
                if compare_rf:
                    f.write("\n\n")
                    f.write("=" * 60 + "\n")
                    f.write("RANDOM FOREST RESULTS:\n")
                    f.write("-" * 60 + "\n")
                    f.write(f"MEAN ACCURACY:    {np.mean(rf_accuracies):.3f}\n")
                    f.write(f"MEAN AUC:         {np.mean(rf_aucs) if rf_aucs else 0.5:.3f} {auc_note}\n")
                    f.write(f"MEAN SENSITIVITY: {np.mean(rf_sensitivities):.3f}\n")
                    f.write(f"MEAN SPECIFICITY: {np.mean(rf_specificities):.3f}\n")

                    f.write("\n")
                    f.write("=" * 60 + "\n")
                    f.write("MODEL COMPARISON:\n")
                    f.write("-" * 60 + "\n")

                    acc_diff = np.mean(lr_accuracies) - np.mean(rf_accuracies)
                    auc_diff = np.mean(lr_aucs) - np.mean(rf_aucs) if (lr_aucs and rf_aucs) else 0
                    sens_diff = np.mean(lr_sensitivities) - np.mean(rf_sensitivities)
                    spec_diff = np.mean(lr_specificities) - np.mean(rf_specificities)

                    f.write(f"Accuracy Difference (LR - RF): {acc_diff:+.3f}\n")
                    f.write(f"AUC Difference (LR - RF):      {auc_diff:+.3f}\n")
                    f.write(f"Sensitivity Difference:        {sens_diff:+.3f}\n")
                    f.write(f"Specificity Difference:        {spec_diff:+.3f}\n")

                    f.write("\n")
                    if abs(acc_diff) < 0.02 and abs(auc_diff) < 0.02:
                        f.write("INTERPRETATION:\n")
                        f.write("Models perform nearly identically, suggesting:\n")
                        f.write("  - Features are highly linearly separable\n")
                        f.write("  - Strong signal-to-noise ratio in biomarkers\n")
                        f.write("  - Simple linear relationships dominate\n")
                        f.write("  - Complex non-linear modeling not needed\n")
                    elif np.mean(lr_accuracies) > np.mean(rf_accuracies):
                        f.write("INTERPRETATION:\n")
                        f.write("Logistic Regression outperforms Random Forest, suggesting:\n")
                        f.write("  - Linear relationships are sufficient\n")
                        f.write("  - RF may be overfitting on small dataset\n")
                        f.write("  - Simpler model is preferred (Occam's Razor)\n")
                    else:
                        f.write("INTERPRETATION:\n")
                        f.write("Random Forest outperforms Logistic Regression, suggesting:\n")
                        f.write("  - Non-linear relationships may exist\n")
                        f.write("  - Feature interactions are important\n")
                        f.write("  - More complex model captures data better\n")

                    f.write("\n")
                    f.write("=" * 60 + "\n")
                    f.write("\nFOLD-BY-FOLD COMPARISON:\n")
                    f.write("-" * 60 + "\n")
                    f.write(f"{'Fold':<8} {'LR Acc':<10} {'RF Acc':<10} {'LR AUC':<10} {'RF AUC':<10}\n")
                    f.write("-" * 60 + "\n")

                    for i in range(len(lr_accuracies)):
                        lr_auc_val = lr_aucs[i] if i < len(lr_aucs) else 0.5
                        rf_auc_val = rf_aucs[i] if i < len(rf_aucs) else 0.5
                        f.write(
                            f"{i + 1:<8} {lr_accuracies[i]:<10.3f} {rf_accuracies[i]:<10.3f} {lr_auc_val:<10.3f} {rf_auc_val:<10.3f}\n")

        # Return statement at the very end
        if compare_rf:
            return np.mean(lr_accuracies), np.mean(rf_accuracies)
        else:
            return np.mean(lr_accuracies), np.mean(lr_accuracies)

    def explain_model(self, df, compare_rf=False):
        """Train final model on full data and generate SHAP explanation for both models

        NOTE: This trains on the FULL dataset for interpretation purposes only.
        This should NOT be used for performance estimation - that's what CV is for.

        Args:
            df: DataFrame with features and labels
            compare_rf: If True, also generate SHAP for Random Forest
        """
        print("\nGenerating AI Explanations (SHAP)...")
        print("⚠️  Training on full dataset for feature interpretation (not for performance evaluation)")
        X, y = self.prepare_data(df)

        # Train LR on full dataset
        self.model.fit(X, y)

        # Train RF on full dataset if comparing
        rf_model = None
        if compare_rf:
            print("Training Random Forest model...")
            from sklearn.ensemble import RandomForestClassifier
            rf_model = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
                ('rf', RandomForestClassifier(
                    class_weight='balanced',
                    random_state=42,
                    n_estimators=100,
                    max_depth=5
                ))
            ])
            rf_model.fit(X, y)
            print("✓ Random Forest trained successfully")

        # Correlation matrix (only need one, not model-specific)
        try:
            plt.figure(figsize=(8, 6))
            corr = X.corr()
            sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f')
            plt.title("Feature Correlation Matrix (Full Dataset)", fontsize=14)
            corr_path = os.path.join(self.output_dir, "feature_correlation.png")
            plt.savefig(corr_path, bbox_inches='tight')
            plt.close()
            print(f"✓ Correlation matrix saved to {corr_path}")
        except Exception as e:
            print(f"Could not generate correlation plot: {e}")

        # ============================================================
        # LOGISTIC REGRESSION SHAP
        # ============================================================
        print("\nGenerating SHAP for Logistic Regression...")

        pipeline_lr = self.model
        preprocessor_lr = Pipeline(pipeline_lr.steps[:-1])
        X_processed_lr = preprocessor_lr.transform(X)
        lr_model = pipeline_lr.named_steps['lr']

        explainer_lr = shap.LinearExplainer(lr_model, X_processed_lr)
        shap_values_raw_lr = explainer_lr.shap_values(X_processed_lr)

        print(f"DEBUG LR: X shape: {X.shape}")
        print(f"DEBUG LR: X_processed shape: {X_processed_lr.shape}")
        print(f"DEBUG LR: shap_values_raw type: {type(shap_values_raw_lr)}")

        if isinstance(shap_values_raw_lr, list) and len(shap_values_raw_lr) == 2:
            base_value_lr = explainer_lr.expected_value[1]
            shap_values_raw_lr = np.array(shap_values_raw_lr[1])
        elif isinstance(shap_values_raw_lr, np.ndarray) and len(shap_values_raw_lr.shape) == 2:
            base_value_lr = explainer_lr.expected_value
        else:
            base_value_lr = explainer_lr.expected_value
            shap_values_raw_lr = np.array(shap_values_raw_lr)

        print(f"DEBUG LR: Final shap_values shape: {shap_values_raw_lr.shape}")

        if hasattr(base_value_lr, "__len__") and len(base_value_lr) > 0:
            if isinstance(base_value_lr, np.ndarray) and len(base_value_lr.shape) > 0:
                base_value_lr = base_value_lr[0]
            elif isinstance(base_value_lr, list):
                base_value_lr = base_value_lr[0]

        exp_lr = shap.Explanation(
            values=shap_values_raw_lr,
            base_values=np.repeat(base_value_lr, len(X)),
            data=X.values,
            feature_names=self.features
        )

        # LR Bar Plot
        plt.figure(figsize=(10, 6))
        shap.plots.bar(exp_lr, max_display=10, show=False)
        plt.title("Logistic Regression: Top Factors (Average Impact)", fontsize=14, pad=15)
        save_path_bar_lr = os.path.join(self.output_dir, "LR_feature_importance.png")
        plt.savefig(save_path_bar_lr, bbox_inches='tight')
        plt.close()
        print(f"✓ LR Feature Importance Chart saved to {save_path_bar_lr}")

        # LR Violin Plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values_raw_lr, X, plot_type="violin", max_display=10, show=False)
        plt.title("LR: Impact Distribution (Spread & Shape)", fontsize=14, pad=15)
        save_path_violin_lr = os.path.join(self.output_dir, "LR_shap_violin.png")
        plt.savefig(save_path_violin_lr, bbox_inches='tight')
        plt.close()
        print(f"✓ LR Violin Plot saved to {save_path_violin_lr}")

        # LR Dot Plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values_raw_lr, X, plot_type="dot", max_display=10, show=False)
        plt.title("LR: Individual Patient Impact (Each Dot is a Person)", fontsize=14, pad=15)
        save_path_dots_lr = os.path.join(self.output_dir, "LR_shap_dotplot.png")
        plt.savefig(save_path_dots_lr, bbox_inches='tight')
        plt.close()
        print(f"✓ LR Dot Plot saved to {save_path_dots_lr}")

        # ============================================================
        # RANDOM FOREST SHAP (if comparing)
        # ============================================================
        if compare_rf and rf_model is not None:
            print("\n" + "=" * 60)
            print("Generating SHAP for Random Forest...")
            print("=" * 60)

            try:
                preprocessor_rf = Pipeline(rf_model.steps[:-1])
                X_processed_rf = preprocessor_rf.transform(X)
                rf_estimator = rf_model.named_steps['rf']

                print(f"Creating TreeExplainer...")
                explainer_rf = shap.TreeExplainer(rf_estimator)
                print(f"Computing SHAP values...")
                shap_values_raw_rf = explainer_rf.shap_values(X_processed_rf)

                # For binary classification, TreeExplainer returns list of 2 arrays
                if isinstance(shap_values_raw_rf, list) and len(shap_values_raw_rf) == 2:
                    shap_values_raw_rf = shap_values_raw_rf[1]  # Positive class (AD)
                    base_value_rf = explainer_rf.expected_value[1]
                else:
                    base_value_rf = explainer_rf.expected_value

                # Ensure base_value is scalar
                if hasattr(base_value_rf, "__len__") and len(base_value_rf) > 0:
                    if isinstance(base_value_rf, np.ndarray):
                        base_value_rf = float(base_value_rf.flatten()[0])
                    elif isinstance(base_value_rf, list):
                        base_value_rf = float(base_value_rf[0])
                else:
                    base_value_rf = float(base_value_rf)

                # Ensure feature_names is a proper list
                feature_names_list = list(self.features)

                # ========================================
                # RF BAR PLOT - MANUAL (shap.plots.bar doesn't work with TreeExplainer)
                # ========================================
                print("Creating RF Bar Plot...")

                # Calculate mean absolute SHAP values for each feature
                #feature_importance = np.abs(shap_values_raw_rf).mean(axis=0)

                if len(shap_values_raw_rf.shape) == 3:
                    feature_importance = np.abs(shap_values_raw_rf[:, :, 1]).mean(axis=0)
                else:
                    feature_importance = np.abs(shap_values_raw_rf).mean(axis=0)

                # Convert to regular Python list to avoid numpy array issues
                feature_importance = feature_importance.tolist()

                # Create a list of (feature_name, importance_value) tuples
                feature_importance_pairs = list(zip(feature_names_list, feature_importance))

                # Sort by importance (descending) - now x[1] is a Python float, not numpy array
                feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)

                # Take top 10 (or all if less than 10)
                top_n = min(10, len(feature_importance_pairs))
                top_features = [pair[0] for pair in feature_importance_pairs[:top_n]]
                top_values = [pair[1] for pair in feature_importance_pairs[:top_n]]

                # Create bar plot
                fig, ax = plt.subplots(figsize=(10, 6))
                y_pos = np.arange(len(top_features))
                colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
                #y_pos = np.arange(len(top_features))

                ax.barh(y_pos, top_values, color=colors)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(top_features)
                ax.invert_yaxis()  # Highest value at top
                ax.set_xlabel('Mean |SHAP value| (Average Impact)', fontsize=11)
                ax.set_title('Random Forest: Top Feature Importance', fontsize=14, fontweight='bold', pad=15)
                ax.grid(axis='x', alpha=0.3)

                plt.tight_layout()
                save_path_bar_rf = os.path.join(self.output_dir, "RF_feature_importance.png")
                plt.savefig(save_path_bar_rf, bbox_inches='tight', dpi=150)
                plt.close(fig)
                print(f"✓ RF Feature Importance Chart saved to {save_path_bar_rf}")
                # ========================================
                # RF VIOLIN PLOT - SAME AS LR
                # ========================================
                print("Creating RF Violin Plot...")

                fig = plt.figure(figsize=(10, 6))
                shap.summary_plot(
                    shap_values_raw_rf,
                    X,
                    plot_type="violin",
                    max_display=10,
                    show=False,
                    feature_names=feature_names_list
                )
                plt.title("RF: Impact Distribution (Spread & Shape)", fontsize=14, pad=15)
                plt.tight_layout()

                save_path_violin_rf = os.path.join(self.output_dir, "RF_shap_violin.png")
                plt.savefig(save_path_violin_rf, bbox_inches='tight', dpi=150)
                plt.close(fig)
                print(f"✓ RF Violin Plot saved to {save_path_violin_rf}")

                # ========================================
                # RF DOT PLOT - SAME AS LR
                # ========================================
                print("Creating RF Dot Plot...")

                fig = plt.figure(figsize=(10, 6))
                shap.summary_plot(
                    shap_values_raw_rf,
                    X,
                    plot_type="dot",
                    max_display=10,
                    show=False,
                    feature_names=feature_names_list
                )
                plt.title("RF: Individual Patient Impact (Each Dot is a Person)", fontsize=14, pad=15)
                plt.tight_layout()

                save_path_dots_rf = os.path.join(self.output_dir, "RF_shap_dotplot.png")
                plt.savefig(save_path_dots_rf, bbox_inches='tight', dpi=150)
                plt.close(fig)
                print(f"✓ RF Dot Plot saved to {save_path_dots_rf}")

                # ========================================
                # SIDE-BY-SIDE COMPARISON (LR vs RF)
                # ========================================
                print("\nGenerating side-by-side comparison...")

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

                # LR on left
                plt.sca(ax1)
                shap.summary_plot(
                    shap_values_raw_lr,
                    X,
                    plot_type="dot",
                    max_display=10,
                    show=False,
                    feature_names=list(self.features)
                )
                ax1.set_title("Logistic Regression", fontsize=15, fontweight='bold')

                # RF on right
                plt.sca(ax2)
                shap.summary_plot(
                    shap_values_raw_rf,
                    X,
                    plot_type="dot",
                    max_display=10,
                    show=False,
                    feature_names=feature_names_list
                )
                ax2.set_title("Random Forest", fontsize=15, fontweight='bold')

                plt.suptitle("Model Comparison: Feature Importance", fontsize=17, fontweight='bold', y=0.995)
                plt.tight_layout(rect=[0, 0, 1, 0.98])

                save_path_comparison = os.path.join(self.output_dir, "LR_vs_RF_comparison.png")
                plt.savefig(save_path_comparison, bbox_inches='tight', dpi=150)
                plt.close(fig)
                print(f"✓ Comparison Plot saved to {save_path_comparison}")

                # ========================================
                # BAR CHART COMPARISON (Feature Importance)
                # ========================================
                print("\nGenerating bar chart comparison...")

                # Calculate mean absolute SHAP for both models
                lr_importance = np.abs(shap_values_raw_lr).mean(axis=0)
                rf_importance = np.abs(shap_values_raw_rf).mean(axis=0)

                # Create comparison bar chart
                fig, ax = plt.subplots(figsize=(12, 6))

                x = np.arange(len(feature_names_list))
                width = 0.35

                bars1 = ax.bar(x - width / 2, lr_importance, width, label='Logistic Regression', color='#3498db',
                               alpha=0.8)
                bars2 = ax.bar(x + width / 2, rf_importance, width, label='Random Forest', color='#e74c3c', alpha=0.8)

                ax.set_xlabel('Features', fontsize=12, fontweight='bold')
                ax.set_ylabel('Mean |SHAP value|', fontsize=12, fontweight='bold')
                ax.set_title('Feature Importance Comparison: LR vs RF', fontsize=14, fontweight='bold', pad=15)
                ax.set_xticks(x)
                ax.set_xticklabels(feature_names_list, rotation=45, ha='right')
                ax.legend(fontsize=11)
                ax.grid(axis='y', alpha=0.3)

                plt.tight_layout()

                save_path_bar_comparison = os.path.join(self.output_dir, "LR_vs_RF_bar_comparison.png")
                plt.savefig(save_path_bar_comparison, bbox_inches='tight', dpi=150)
                plt.close(fig)
                print(f"✓ Bar Comparison Plot saved to {save_path_bar_comparison}")

            except Exception as e:
                print(f"❌ ERROR generating RF SHAP: {e}")
                import traceback
                traceback.print_exc()

        elif compare_rf and rf_model is None:
            print("⚠️  WARNING: compare_rf=True but rf_model was not trained!")

        print("\n✅ SHAP explanation generation complete")


def plot_diagnostic_psd_shift(extractor, data_dir, output_dir):
    """Generates a comparison plot showing Alpha Slowing in AD."""
    print("\n📈 Generating Diagnostic PSD Visualization (Slowing Analysis)...")

    ad_sub = "sub-001"
    hc_sub = "sub-037"

    f_ad, p_ad = extractor.get_subject_psd(ad_sub, data_dir)
    f_hc, p_hc = extractor.get_subject_psd(hc_sub, data_dir)

    if f_ad is None or f_hc is None:
        print("    ! Could not generate PSD plot: Data extraction failed.")
        return

    plt.figure(figsize=(10, 6))

    p_ad = np.atleast_1d(p_ad)
    p_hc = np.atleast_1d(p_hc)
    p_ad[p_ad <= 0] = 1e-12
    p_hc[p_hc <= 0] = 1e-12

    p_ad_pl = 10 * np.log10(p_ad)
    p_hc_pl = 10 * np.log10(p_hc)

    # Focus on 1-25 Hz for best visualization
    mask = (f_ad >= 1) & (f_ad <= 25)

    f_plot = f_ad[mask]
    p_hc_plot = p_hc_pl[mask]
    p_ad_plot = p_ad_pl[mask]

    plt.plot(f_plot, p_hc_plot, color='#2E86C1', lw=3, label='Healthy Control (HC)')
    plt.plot(f_plot, p_ad_plot, color='#C0392B', lw=3, label="Alzheimer's Patient (AD)")

    # Highlight Alpha Band
    plt.axvspan(8, 12, color='gray', alpha=0.1, label='Alpha Band (8-12 Hz)')

    # Highlight Peaks (Slowing)
    alpha_mask = (f_plot >= 7) & (f_plot <= 13)
    if np.any(alpha_mask):
        paf_hc = f_plot[alpha_mask][np.argmax(p_hc_plot[alpha_mask])]
        idx_hc = np.argmin(np.abs(f_plot - paf_hc))
        plt.annotate('HC Peak', xy=(paf_hc, p_hc_plot[idx_hc]), xytext=(paf_hc + 2, p_hc_plot[idx_hc] + 2),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))

        paf_ad = f_plot[alpha_mask][np.argmax(p_ad_plot[alpha_mask])]
        idx_ad = np.argmin(np.abs(f_plot - paf_ad))
        plt.annotate('AD Peak (Shifted Left)', xy=(paf_ad, p_ad_plot[idx_ad]),
                     xytext=(paf_ad - 4, p_ad_plot[idx_ad] + 4),
                     arrowprops=dict(facecolor='red', shrink=0.05, width=1, headwidth=5))

    plt.title("The 'Slowing' Biomarker: PSD Shift in Alzheimer's", fontsize=15, fontweight='bold', pad=20)

    plt.xlabel("Frequency (Hz)", fontsize=12)
    plt.ylabel("Power (dB/Hz)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=True, fontsize=10)

    sns.despine()
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "diagnostic_psd_slowing.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"✅ Diagnostic PSD plot saved to: {plot_path}")


def main():
    # Parse command line arguments for flexibility
    import gc
    import pandas as pd
    import numpy as np
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
    data_dir_ec = "C:\\Users\\amiri\\OneDrive\Documents\SrivibhavBCIProject\Alzheimers_Detection\\alz_data\script_downloaded_open_neuro\eeg_alzheimer_data\\raw\ds004504"
    data_dir_eo = "C:\\Users\\amiri\\OneDrive\Documents\SrivibhavBCIProject\Alzheimers_Detection\\alz_data\script_downloaded_open_neuro\eeg_alzheimer_data_eyes_open\\raw\ds006036"

    print(f"\n{'=' * 50}")
    print(f"Starting Multi-Condition Extraction Analysis (EC + EO)")
    if TINY_MODE:
        print(f"Mode: TINY TEST (EC + EO Fusion)")
    else:
        print(f"Mode: {'TEST' if TEST_MODE else 'FULL'} (EC + EO Fusion)")
    print(f"{'=' * 50}\n")

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

        print(f"[{i + 1}/{len(participants)}] Processing {sub_id} (Group: {group})...")

        result_entry = {
            'participant_id': sub_id,
            'Group': group,
            'Age': row.get('Age', np.nan)
        }

        # 1. Extract Eyes Closed (EC) Regional Features
        print(f"    -> Extracting Posterior EC biomarkers...")
        try:
            ec_pli = extractor.compute_subject_pli(sub_id, data_dir_ec, test_mode=TEST_MODE, cluster_only=True)
            if ec_pli:
                result_entry['EC_Posterior_PLI'] = ec_pli['Global_PLI']

            ec_spectral = extractor.compute_subject_spectral_features(sub_id, data_dir_ec, test_mode=TEST_MODE, cluster_only=True)
            if ec_spectral:
                result_entry['EC_Posterior_Theta_Alpha_Ratio'] = ec_spectral.get('Theta_Alpha_Ratio')
                result_entry['EC_Posterior_Rel_Alpha'] = ec_spectral.get('Rel_Alpha')
                result_entry['EC_Posterior_Rel_High_Alpha'] = ec_spectral.get('Rel_High_Alpha')
                result_entry['EC_Posterior_Peak_Alpha_Freq'] = ec_spectral.get('Peak_Alpha_Freq')
                # Store absolute power for diagnostic check
                result_entry['EC_Abs_Alpha'] = ec_spectral.get('Abs_Alpha', np.nan)

        except Exception as e:
            print(f"    ! Error extracting EC data for {sub_id}: {e}")

        # 2. Extract Eyes Open (EO) Regional Features
        print(f"    -> Extracting Posterior EO biomarkers...")
        try:
            eo_pli = extractor.compute_subject_pli(sub_id, data_dir_eo, test_mode=TEST_MODE, cluster_only=True)
            if eo_pli:
                result_entry['EO_Posterior_PLI'] = eo_pli['Global_PLI']

            eo_spectral = extractor.compute_subject_spectral_features(sub_id, data_dir_eo, test_mode=TEST_MODE, cluster_only=True)
            if eo_spectral:
                result_entry['EO_Posterior_Rel_Alpha'] = eo_spectral.get('Rel_Alpha', np.nan)
                # Store absolute power for ABR calc
                result_entry['EO_Abs_Alpha'] = eo_spectral.get('Abs_Alpha', np.nan)

        except Exception as e:
            print(f"    ! Note: EO data missing/failed for {sub_id}")

        # 3. Calculate Regional "Reactivity" (Robust Metrics)
        if result_entry.get('EC_Abs_Alpha') and result_entry.get('EO_Abs_Alpha'):
            ec_alpha = result_entry['EC_Abs_Alpha']
            eo_alpha = result_entry['EO_Abs_Alpha']
            if (ec_alpha + eo_alpha) > 0:
                # 3a. Normalized Alpha Suppression Index (NASI)
                # Bounded between -1 and 1. Higher value = Better suppression (Lower EO Alpha)
                result_entry['Alpha_Reactivity_Normalized'] = (ec_alpha - eo_alpha) / (ec_alpha + eo_alpha)

                # 3b. Log-Ratio (dB Scale)
                # More linear for statistical tests
                if eo_alpha > 0:
                    result_entry['Alpha_Reactivity_Log_Ratio'] = 10 * np.log10(ec_alpha / eo_alpha)

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

        print(f"    {'-' * 40}")

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
    print("\n" + "=" * 60)
    print("STARTING MODEL TRAINING")
    print("=" * 60)

    # 1. Create timestamped directory FIRST
    timestamp = datetime.now().strftime("%b%d_%H%M")
    temp_mode_str = "Tiny" if TINY_MODE else ("Test" if TEST_MODE else "Full")
    output_dir = f"run_v6_LR_{timestamp}_{temp_mode_str}"

    print(f"\n📁 Creating output directory: {output_dir}")
    print(f"📁 Current working directory: {os.getcwd()}")

    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"✓ Directory created: {output_dir}")
        else:
            print(f"⚠️  Directory already exists: {output_dir}")

        # Verify it was created
        if os.path.exists(output_dir):
            print(f"✓ Directory verified: {output_dir}")
            print(f"✓ Directory is writable: {os.access(output_dir, os.W_OK)}")
        else:
            print(f"❌ ERROR: Directory was not created!")
            return

    except Exception as e:
        print(f"❌ ERROR creating directory: {e}")
        import traceback
        traceback.print_exc()
        return

    classifier = AlzheimerClassifier(output_dir=output_dir)
    print(f"✓ Classifier initialized with output_dir: {classifier.output_dir}")

    # 2. Run Robust Cross-Validation with RF comparison
    if len(df_results) < 35:
        print(
            f"\n[Auto-Switch] Dataset too small for 5-Fold (N={len(df_results)}). Switching to Leave-One-Out CV for stability.")
        use_loocv_flag = True
    else:
        use_loocv_flag = False

    print("\n" + "=" * 60)
    print("RUNNING CROSS-VALIDATION")
    print("=" * 60)

    # Modified to return both accuracies
    try:
        result = classifier.train_with_cv(df_results, use_loocv=use_loocv_flag, compare_rf=True)
        print(f"\n✓ train_with_cv returned: {result}")
        print(f"✓ Result type: {type(result)}")

        if isinstance(result, tuple) and len(result) == 2:
            mean_acc_lr, mean_acc_rf = result
            print(f"✓ LR Accuracy: {mean_acc_lr:.3f}")
            print(f"✓ RF Accuracy: {mean_acc_rf:.3f}")
        else:
            print(f"❌ ERROR: train_with_cv returned unexpected format: {result}")
            mean_acc_lr = result if isinstance(result, (int, float)) else 0.5
            mean_acc_rf = mean_acc_lr

    except Exception as e:
        print(f"❌ ERROR in train_with_cv: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. Finalize folder name with BOTH accuracies
    print("\n" + "=" * 60)
    print("RENAMING OUTPUT DIRECTORY")
    print("=" * 60)

    final_output_dir = f"{output_dir}_Acc{int(mean_acc_lr * 100)}_{int(mean_acc_rf * 100)}"
    print(f"📁 Original directory: {output_dir}")
    print(f"📁 Target directory: {final_output_dir}")

    try:
        # Check if original directory still exists
        if not os.path.exists(output_dir):
            print(f"⚠️  WARNING: Original directory doesn't exist: {output_dir}")
            print(f"   Looking for directories starting with 'run_v6_LR'...")
            for item in os.listdir('.'):
                if item.startswith('run_v6_LR') and timestamp in item:
                    print(f"   Found: {item}")
                    output_dir = item
                    final_output_dir = f"{output_dir}_Acc{int(mean_acc_lr * 100)}_{int(mean_acc_rf * 100)}"
                    break

        # Check if target already exists
        if os.path.exists(final_output_dir):
            print(f"⚠️  Target directory already exists: {final_output_dir}")
            print(f"   Using existing directory...")
        else:
            # Rename
            os.rename(output_dir, final_output_dir)
            print(f"✓ Directory renamed successfully")

        # Verify the final directory
        if os.path.exists(final_output_dir):
            print(f"✓ Final directory verified: {final_output_dir}")
            classifier.output_dir = final_output_dir
        else:
            print(f"❌ ERROR: Final directory doesn't exist after rename!")
            print(f"   Falling back to original: {output_dir}")
            final_output_dir = output_dir
            classifier.output_dir = output_dir

    except Exception as e:
        print(f"❌ ERROR renaming directory: {e}")
        print(f"   Falling back to original: {output_dir}")
        final_output_dir = output_dir
        classifier.output_dir = output_dir
        import traceback
        traceback.print_exc()

    print(f"\n📂 Final Results Directory: {final_output_dir}")
    print(f"📂 Full Path: {os.path.abspath(final_output_dir)}")

    # 6. AI Explanations
    print("\n" + "🎯" * 30)
    print("🎯 CHECKPOINT: About to call explain_model")
    print(f"🎯 classifier object: {classifier}")
    print(f"🎯 df_results shape: {df_results.shape}")
    print(f"🎯 final_output_dir: {final_output_dir}")
    print("🎯" * 30 + "\n")

    try:
        print("CALLING: classifier.explain_model(df_results, compare_rf=True)")
        classifier.explain_model(df_results, compare_rf=True)
        print("\n🎯 explain_model() COMPLETED SUCCESSFULLY")
    except Exception as e:
        print(f"\n❌ ERROR in explain_model: {e}")
        import traceback
        traceback.print_exc()

    # Generate the Comparison Plot for the User
    print("\n🎯 CHECKPOINT: About to generate PSD plot")
    try:
        plot_diagnostic_psd_shift(extractor, data_dir_ec, final_output_dir)
        print("✓ PSD plot completed")
    except Exception as e:
        print(f"❌ ERROR in PSD plot: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n✅ REFINED ANALYSIS COMPLETE")
    print(f"All records and premium charts stored in: {classifier.output_dir}")

    print("\n" + "📁" * 30)
    print("📁 FILES IN OUTPUT DIRECTORY:")
    print("📁" * 30)
    for file in os.listdir(final_output_dir):
        print(f"  - {file}")
    print("📁" * 30)


if __name__ == "__main__":
    main()