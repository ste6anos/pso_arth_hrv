import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import base64
from io import BytesIO
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# ===========================================================================
# CONFIGURATION
# ===========================================================================

APPLY_VIF = False            # VIF-based collinearity filtering
APPLY_SCALING = True        # StandardScaler (recommended for GPR)
APPLY_CAPPING = True        # Outlier capping (Z=3)
VIF_THRESHOLD = 10.0         # Collinearity threshold

# Covariates to include ["DEMOGR_AGE", "DEMOGR_SEX", "PAT_SMOKE_PAST_H", "DOC_PAT_BMI", "CRP_mg_dL"]
COVARIATES = ["DEMOGR_AGE", "PAT_SMOKE_PAST_H", "CRP_mg_dL"]


# ===========================================================================
# HELPER FUNCTIONS
# ===========================================================================

def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string."""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64


def calculate_vif(df):
    """Calculate VIF for each feature to detect multicollinearity."""
    vif_data = pd.DataFrame()
    vif_data["Feature"] = df.columns
    vif_data["VIF"] = [variance_inflation_factor(df.values, i) for i in range(len(df.columns))]
    return vif_data.sort_values('VIF', ascending=False)


def remove_collinear_features(df, threshold=5.0):
    """
    Remove features with high VIF (multicollinearity).
    VIF > 5: moderate collinearity
    VIF > 10: high collinearity
    """
    print(f"\n{'='*60}")
    print("COLLINEARITY ANALYSIS (VIF)")
    print(f"{'='*60}")

    features = df.copy()
    removed_features = []

    while True:
        vif_data = calculate_vif(features)
        print(f"\n{vif_data.to_string(index=False)}")

        max_vif = vif_data['VIF'].max()

        if max_vif > threshold:
            feature_to_remove = vif_data.loc[vif_data['VIF'].idxmax(), 'Feature']
            print(f"\nRemoving '{feature_to_remove}' (VIF={max_vif:.2f})")
            features = features.drop(columns=[feature_to_remove])
            removed_features.append(feature_to_remove)
        else:
            break

    print(f"\nFinal features: {list(features.columns)}")
    if removed_features:
        print(f"Removed features: {removed_features}")

    return features, removed_features


def cap_outliers(df, z_threshold=3):
    """Cap outliers using z-score method."""
    df_capped = df.copy()

    for col in df_capped.columns:
        mean = df_capped[col].mean()
        std = df_capped[col].std()

        if std > 0:
            upper_limit = mean + z_threshold * std
            lower_limit = mean - z_threshold * std

            n_capped = ((df_capped[col] > upper_limit) | (df_capped[col] < lower_limit)).sum()

            if n_capped > 0:
                df_capped[col] = df_capped[col].clip(lower_limit, upper_limit)
                print(f"  {col}: Capped {n_capped} outliers (±{z_threshold}σ)")

    return df_capped


def preprocess_data(X, y, covariates_df, apply_vif=True, apply_scaling=True, apply_capping=True, vif_threshold=5.0):
    """
    Preprocess features: Merge covariates, VIF removal, capping, scaling.

    Parameters:
    - X: HRV features DataFrame
    - y: Target series
    - covariates_df: Covariates DataFrame
    - apply_vif: Whether to apply VIF-based collinearity filtering
    - apply_scaling: Whether to apply StandardScaler
    - apply_capping: Whether to cap outliers with z-score
    - vif_threshold: VIF threshold for collinearity removal
    """
    print(f"\n{'='*60}")
    print("PREPROCESSING PIPELINE")
    print(f"{'='*60}")
    print(f"Initial HRV features: {X.shape[1]}")
    print(f"Covariates: {len(COVARIATES)}")
    print(f"Samples: {X.shape[0]}")

    # Merge HRV features with covariates
    X_combined = pd.concat([X, covariates_df], axis=1)
    print(f"\nCombined features (HRV + Covariates): {X_combined.shape[1]}")

    # Convert categorical columns to numeric
    for col in X_combined.columns:
        if X_combined[col].dtype == 'object' or X_combined[col].dtype.name == 'category':
            print(f"\nConverting categorical column '{col}' to numeric...")
            X_combined[col] = pd.Categorical(X_combined[col]).codes
            print(f"  Unique values: {X_combined[col].nunique()}")

    # Step 1: Remove collinear features (optional)
    X_reduced = X_combined.copy()
    removed_features = []

    if apply_vif:
        X_reduced, removed_features = remove_collinear_features(X_reduced, threshold=vif_threshold)
        print(f"\nAfter VIF filtering: {X_reduced.shape[1]} features")
    else:
        print("\nVIF filtering: DISABLED (keeping all features)")

    # Step 2: Cap outliers (optional)
    if apply_capping:
        print(f"\n{'='*60}")
        print("OUTLIER CAPPING (Z-score method, threshold=3)")
        print(f"{'='*60}")
        X_capped = cap_outliers(X_reduced, z_threshold=3)
    else:
        print("\nOutlier capping: DISABLED")
        X_capped = X_reduced.copy()

    # Step 3: Scaling (optional but recommended for GPR)
    scaler = None
    if apply_scaling:
        print(f"\n{'='*60}")
        print("SCALING (StandardScaler)")
        print(f"{'='*60}")
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X_capped),
            columns=X_capped.columns,
            index=X_capped.index
        )
        print("Features scaled to mean=0, std=1")
    else:
        print("\nScaling: DISABLED")
        X_scaled = X_capped.copy()

    print(f"\n{'='*60}")
    print("PREPROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Final features: {X_scaled.shape[1]}")
    print(f"Final samples: {X_scaled.shape[0]}")

    return X_scaled, removed_features, scaler


# ===========================================================================
# HTML REPORT GENERATION
# ===========================================================================

def create_html_report(results, output_file='gpr_report_dual_state_90m.html'):
    """Generate comprehensive HTML report with dual-state GPR results."""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gaussian Process Regression - Dual State Analysis</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', sans-serif;
            line-height: 1.6;
            color: #24292e;
            background: #f8f9fa;
        }}

        .container {{
            max-width: 1600px;
            margin: 0 auto;
            background: white;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}

        .header {{
            background: white;
            border-bottom: 2px solid #e1e4e8;
            padding: 40px;
            text-align: center;
        }}

        .header h1 {{
            font-size: 2em;
            margin-bottom: 8px;
            font-weight: 600;
            color: #24292e;
        }}

        .header p {{
            font-size: 1em;
            color: #586069;
        }}

        .timestamp {{
            font-size: 0.85em;
            color: #6a737d;
            margin-top: 8px;
        }}

        .content {{
            padding: 40px;
        }}

        .config-section {{
            background: #fafbfc;
            border: 1px solid #e1e4e8;
            border-radius: 4px;
            padding: 20px;
            margin-bottom: 40px;
        }}

        .config-section h3 {{
            font-size: 1.125em;
            font-weight: 600;
            margin-bottom: 16px;
            color: #24292e;
        }}

        .config-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 12px;
        }}

        .config-item {{
            padding: 8px 12px;
            background: white;
            border: 1px solid #e1e4e8;
            border-radius: 3px;
            font-size: 0.9em;
        }}

        .config-item strong {{
            color: #24292e;
        }}

        .summary-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 16px;
            margin-bottom: 40px;
        }}

        .stat-card {{
            background: white;
            padding: 20px;
            border: 1px solid #e1e4e8;
            border-radius: 4px;
            text-align: center;
        }}

        .stat-card h3 {{
            font-size: 0.875em;
            color: #586069;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-weight: 600;
        }}

        .stat-card p {{
            font-size: 1.75em;
            font-weight: 600;
            color: #2c3e50;
        }}

        .target-section {{
            margin-bottom: 60px;
            padding: 30px;
            background: #fafbfc;
            border: 1px solid #e1e4e8;
            border-radius: 4px;
        }}

        .target-section h2 {{
            color: #24292e;
            margin-bottom: 24px;
            font-size: 1.75em;
            font-weight: 600;
            padding-bottom: 8px;
            border-bottom: 1px solid #e1e4e8;
        }}

        .state-comparison {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
            margin-top: 20px;
        }}

        .state-panel {{
            background: white;
            border: 1px solid #e1e4e8;
            border-radius: 4px;
            padding: 20px;
        }}

        .state-panel h3 {{
            font-size: 1.25em;
            font-weight: 600;
            margin-bottom: 16px;
            color: #24292e;
            padding-bottom: 8px;
            border-bottom: 1px solid #e1e4e8;
        }}

        .badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 3px;
            font-size: 0.75em;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-left: 8px;
        }}

        .badge-excellent {{ background: #d4edda; color: #155724; }}
        .badge-good {{ background: #d1ecf1; color: #0c5460; }}
        .badge-moderate {{ background: #fff3cd; color: #856404; }}
        .badge-poor {{ background: #f8d7da; color: #721c24; }}

        .info-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 12px;
            margin-bottom: 20px;
        }}

        .info-item {{
            padding: 10px;
            background: #f6f8fa;
            border-left: 3px solid #0366d6;
            font-size: 0.9em;
        }}

        .info-item strong {{
            color: #24292e;
            display: block;
            margin-bottom: 4px;
        }}

        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 12px;
            margin: 20px 0;
        }}

        .metric-box {{
            padding: 12px;
            background: #f6f8fa;
            border: 1px solid #e1e4e8;
            border-radius: 3px;
        }}

        .metric-box h4 {{
            font-size: 0.75em;
            color: #586069;
            margin-bottom: 4px;
            text-transform: uppercase;
            letter-spacing: 0.3px;
        }}

        .metric-box p {{
            font-size: 1.5em;
            font-weight: 600;
            color: #24292e;
        }}

        .plot-container {{
            margin: 20px 0;
            text-align: center;
        }}

        .plot-container img {{
            max-width: 100%;
            border: 1px solid #e1e4e8;
            border-radius: 4px;
        }}

        .features-list {{
            margin: 16px 0;
            padding: 12px;
            background: #f6f8fa;
            border-left: 3px solid #28a745;
            font-size: 0.9em;
        }}

        .features-list strong {{
            display: block;
            margin-bottom: 8px;
            color: #24292e;
        }}

        .features-list ul {{
            margin-left: 20px;
            color: #586069;
        }}

        .removed-features {{
            margin: 16px 0;
            padding: 12px;
            background: #fff3cd;
            border-left: 3px solid #ffc107;
            font-size: 0.9em;
        }}

        .removed-features strong {{
            display: block;
            margin-bottom: 8px;
            color: #856404;
        }}

        .footer {{
            background: #f6f8fa;
            color: #586069;
            padding: 24px;
            text-align: center;
            border-top: 1px solid #e1e4e8;
            font-size: 0.875em;
        }}

        @media (max-width: 1200px) {{
            .state-comparison {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Gaussian Process Regression - Dual State Analysis</h1>
            <p>HRV Metrics Predictive Modeling with Covariates: Low vs High HR Activity</p>
            <p class="timestamp">Generated: {timestamp}</p>
        </div>

        <div class="content">
            <!-- CONFIGURATION -->
            <div class="config-section">
                <h3>Preprocessing Configuration</h3>
                <div class="config-grid">
                    <div class="config-item">
                        <strong>VIF Filtering:</strong> {'Enabled' if APPLY_VIF else 'Disabled'}
                    </div>
                    <div class="config-item">
                        <strong>Outlier Capping:</strong> {'Enabled (Z=3)' if APPLY_CAPPING else 'Disabled'}
                    </div>
                    <div class="config-item">
                        <strong>Scaling:</strong> {'StandardScaler' if APPLY_SCALING else 'Disabled'}
                    </div>
                    <div class="config-item">
                        <strong>VIF Threshold:</strong> {VIF_THRESHOLD}
                    </div>
                </div>
                <div style="margin-top: 16px; padding: 12px; background: white; border: 1px solid #e1e4e8; border-radius: 3px;">
                    <strong style="display: block; margin-bottom: 8px; color: #24292e;">Covariates Included:</strong>
                    <p style="color: #586069; font-size: 0.9em;">{', '.join(COVARIATES)}</p>
                </div>
            </div>

            <!-- SUMMARY STATISTICS -->
            <div class="summary-stats">
                <div class="stat-card">
                    <h3>Total Targets</h3>
                    <p>{len(results)}</p>
                </div>
                <div class="stat-card">
                    <h3>Low HR Avg R²</h3>
                    <p>{np.mean([r['low_hr']['test_r2'] for r in results.values() if 'low_hr' in r]):.3f}</p>
                </div>
                <div class="stat-card">
                    <h3>High HR Avg R²</h3>
                    <p>{np.mean([r['high_hr']['test_r2'] for r in results.values() if 'high_hr' in r]):.3f}</p>
                </div>
                <div class="stat-card">
                    <h3>Best Low HR R²</h3>
                    <p>{max([r['low_hr']['test_r2'] for r in results.values() if 'low_hr' in r]):.3f}</p>
                </div>
                <div class="stat-card">
                    <h3>Best High HR R²</h3>
                    <p>{max([r['high_hr']['test_r2'] for r in results.values() if 'high_hr' in r]):.3f}</p>
                </div>
            </div>
"""

    # Add sections for each target
    for target, state_results in results.items():
        low_hr = state_results.get('low_hr', {})
        high_hr = state_results.get('high_hr', {})

        html += f"""
            <div class="target-section">
                <h2>{target}</h2>

                <div class="state-comparison">
                    <!-- LOW HR ACTIVITY -->
                    <div class="state-panel">
                        <h3>Low HR Activity (Sleep)"""

        # Add badge for Low HR
        if low_hr.get('test_r2', 0) >= 0.7:
            html += '<span class="badge badge-excellent">Excellent</span>'
        elif low_hr.get('test_r2', 0) >= 0.5:
            html += '<span class="badge badge-good">Good</span>'
        elif low_hr.get('test_r2', 0) >= 0.3:
            html += '<span class="badge badge-moderate">Moderate</span>'
        else:
            html += '<span class="badge badge-poor">Poor</span>'

        html += """
                        </h3>

                        <div class="info-grid">
                            <div class="info-item">
                                <strong>Samples</strong>
                                """ + f"{low_hr.get('n_samples', 'N/A')}" + """
                            </div>
                            <div class="info-item">
                                <strong>Features Used</strong>
                                """ + f"{len(low_hr.get('features_used', []))}" + """
                            </div>
                        </div>
"""

        if low_hr.get('features_used'):
            html += f"""
                        <div class="features-list">
                            <strong>Features Used:</strong>
                            <ul>
                                {''.join([f'<li>{f}</li>' for f in low_hr['features_used']])}
                            </ul>
                        </div>
"""

        if low_hr.get('features_removed'):
            html += f"""
                        <div class="removed-features">
                            <strong>Features Removed (VIF):</strong>
                            {', '.join(low_hr['features_removed'])}
                        </div>
"""

        html += """
                        <div class="metrics-grid">
                            <div class="metric-box">
                                <h4>Train R²</h4>
                                <p>""" + f"{low_hr.get('train_r2', 0):.4f}" + """</p>
                            </div>
                            <div class="metric-box">
                                <h4>Test R²</h4>
                                <p>""" + f"{low_hr.get('test_r2', 0):.4f}" + """</p>
                            </div>
                            <div class="metric-box">
                                <h4>Train RMSE</h4>
                                <p>""" + f"{low_hr.get('train_rmse', 0):.4f}" + """</p>
                            </div>
                            <div class="metric-box">
                                <h4>Test RMSE</h4>
                                <p>""" + f"{low_hr.get('test_rmse', 0):.4f}" + """</p>
                            </div>
                            <div class="metric-box">
                                <h4>Train MAE</h4>
                                <p>""" + f"{low_hr.get('train_mae', 0):.4f}" + """</p>
                            </div>
                            <div class="metric-box">
                                <h4>Test MAE</h4>
                                <p>""" + f"{low_hr.get('test_mae', 0):.4f}" + """</p>
                            </div>
                        </div>
"""

        if low_hr.get('plot'):
            html += f"""
                        <div class="plot-container">
                            <img src="data:image/png;base64,{low_hr['plot']}" alt="Low HR Activity Predictions">
                        </div>
"""

        html += """
                    </div>

                    <!-- HIGH HR ACTIVITY -->
                    <div class="state-panel">
                        <h3>High HR Activity (Awake)"""

        # Add badge for High HR
        if high_hr.get('test_r2', 0) >= 0.7:
            html += '<span class="badge badge-excellent">Excellent</span>'
        elif high_hr.get('test_r2', 0) >= 0.5:
            html += '<span class="badge badge-good">Good</span>'
        elif high_hr.get('test_r2', 0) >= 0.3:
            html += '<span class="badge badge-moderate">Moderate</span>'
        else:
            html += '<span class="badge badge-poor">Poor</span>'

        html += """
                        </h3>

                        <div class="info-grid">
                            <div class="info-item">
                                <strong>Samples</strong>
                                """ + f"{high_hr.get('n_samples', 'N/A')}" + """
                            </div>
                            <div class="info-item">
                                <strong>Features Used</strong>
                                """ + f"{len(high_hr.get('features_used', []))}" + """
                            </div>
                        </div>
"""

        if high_hr.get('features_used'):
            html += f"""
                        <div class="features-list">
                            <strong>Features Used:</strong>
                            <ul>
                                {''.join([f'<li>{f}</li>' for f in high_hr['features_used']])}
                            </ul>
                        </div>
"""

        if high_hr.get('features_removed'):
            html += f"""
                        <div class="removed-features">
                            <strong>Features Removed (VIF):</strong>
                            {', '.join(high_hr['features_removed'])}
                        </div>
"""

        html += """
                        <div class="metrics-grid">
                            <div class="metric-box">
                                <h4>Train R²</h4>
                                <p>""" + f"{high_hr.get('train_r2', 0):.4f}" + """</p>
                            </div>
                            <div class="metric-box">
                                <h4>Test R²</h4>
                                <p>""" + f"{high_hr.get('test_r2', 0):.4f}" + """</p>
                            </div>
                            <div class="metric-box">
                                <h4>Train RMSE</h4>
                                <p>""" + f"{high_hr.get('train_rmse', 0):.4f}" + """</p>
                            </div>
                            <div class="metric-box">
                                <h4>Test RMSE</h4>
                                <p>""" + f"{high_hr.get('test_rmse', 0):.4f}" + """</p>
                            </div>
                            <div class="metric-box">
                                <h4>Train MAE</h4>
                                <p>""" + f"{high_hr.get('train_mae', 0):.4f}" + """</p>
                            </div>
                            <div class="metric-box">
                                <h4>Test MAE</h4>
                                <p>""" + f"{high_hr.get('test_mae', 0):.4f}" + """</p>
                            </div>
                        </div>
"""

        if high_hr.get('plot'):
            html += f"""
                        <div class="plot-container">
                            <img src="data:image/png;base64,{high_hr['plot']}" alt="High HR Activity Predictions">
                        </div>
"""

        html += """
                    </div>
                </div>
            </div>
"""

    html += """
        </div>

        <div class="footer">
            <p>Gaussian Process Regression Analysis | HRV Metrics with Covariates</p>
            <p style="margin-top: 8px;">Low HR Activity (Sleep) | High HR Activity (Awake)</p>
        </div>
    </div>
</body>
</html>
"""

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"\n{'='*60}")
    print(f"HTML report saved to '{output_file}'")
    print(f"{'='*60}")


# ===========================================================================
# MAIN EXECUTION
# ===========================================================================

if __name__ == "__main__":
    print("="*60)
    print("GAUSSIAN PROCESS REGRESSION - DUAL STATE ANALYSIS")
    print("="*60)
    print(f"VIF Filtering: {APPLY_VIF}")
    print(f"Scaling: {APPLY_SCALING}")
    print(f"Capping: {APPLY_CAPPING}")
    print(f"VIF Threshold: {VIF_THRESHOLD}")
    print(f"Covariates: {', '.join(COVARIATES)}")
    print("="*60)

    # Load data
    print("\nLoading data...")

    # Load HRV data for both states (they already have patientid column)
    X_df_aw = pd.read_csv("window90m_hrv_mean_measurements_aw.csv")
    X_df_sl = pd.read_csv("window90m_hrv_mean_measurements_sl.csv")

    # Keep only selected HRV metrics: SDNN, ULF, LFHF, VHF
    selected_hrv_metrics = ['HRV_SDNN', 'HRV_LFHF', 'patientid']
    X_df_aw = X_df_aw[selected_hrv_metrics]
    X_df_sl = X_df_sl[selected_hrv_metrics]

    # Load clinical data (targets)
    clinical_data_df = pd.read_csv(r"C:\Users\spbtu\Documents\dataset_psath\bbi_metadata\clinical_d32_T0_v2.csv")

    print(f"  High HR Activity data: {X_df_aw.shape}")
    print(f"  Low HR Activity data: {X_df_sl.shape}")
    print(f"  Clinical data: {clinical_data_df.shape}")

    # Define clinical targets (excluding covariates)
    y_df = clinical_data_df[["patientid", "DOC_PAT_BMI", "DAPSA_Score", "PASDAS",
                              "PSAID_Final_Score", "Overall HAQ Score", "DOC_VAS_H"]]

    # Prepare covariates DataFrame
    covariates_clinical = clinical_data_df[["patientid"] + COVARIATES].copy()

    print(f"  Targets: {y_df.shape[1] - 1}")  # -1 for patientid
    print(f"  Covariates: {len(COVARIATES)}")

    # Store results
    results = {}

    # Process each target
    for target in y_df.columns[1:]:  # Skip patientid
        print(f"\n{'='*60}")
        print(f"TARGET: {target}")
        print(f"{'='*60}")

        # Initialize nested results structure
        results[target] = {
            'low_hr': {},
            'high_hr': {}
        }

        # Process both states
        for state_name, X_df in [('low_hr', X_df_sl), ('high_hr', X_df_aw)]:
            state_label = 'Low HR Activity' if state_name == 'low_hr' else 'High HR Activity'

            print(f"\n--- Processing {state_label} ---")

            # Prepare data for this state
            # X_df already has patientid column
            # Prepare target data
            temp_target = y_df[['patientid', target]]

            # Merge HRV with target
            df = pd.merge(X_df, temp_target, on="patientid", how="inner")

            # Merge with covariates
            df = pd.merge(df, covariates_clinical, on="patientid", how="inner", suffixes=('', '_cov'))

            # Determine which covariates to use (exclude target if it's in covariates)
            covariates_to_use = [c if c != target else c + '_cov' for c in COVARIATES]
            covariates_to_use = [c for c in covariates_to_use if c in df.columns]

            # Drop rows with missing target or covariates
            check_cols = [target] + covariates_to_use
            df = df.dropna(subset=check_cols)

            # Drop patientid column now
            df = df.drop(columns=['patientid'])

            if len(df) < 10:
                print(f"  ⚠️  Insufficient samples ({len(df)}) for {target} - {state_label}")
                continue

            # Separate features and target
            X_raw = df.drop(columns=[target])
            y = df[target].values

            # Split covariates for preprocessing
            hrv_features = [col for col in X_raw.columns if col.startswith('HRV_')]
            covariate_features = [col for col in X_raw.columns if col in covariates_to_use]

            X_hrv = X_raw[hrv_features]
            X_covariates = X_raw[covariate_features]

            # Preprocess data (HRV + covariates together)
            X_processed, removed_features, scaler = preprocess_data(
                X_hrv, y, X_covariates,
                apply_vif=APPLY_VIF,
                apply_scaling=APPLY_SCALING,
                apply_capping=APPLY_CAPPING,
                vif_threshold=VIF_THRESHOLD
            )

            final_variable_names = list(X_processed.columns)

            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y, test_size=0.2, random_state=42
            )

            # Scale target variable (y)
            y_scaler = StandardScaler()
            y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
            y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).ravel()

            # Define GPR kernel
            kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))

            # Train GPR model
            print(f"\n{'='*60}")
            print(f"TRAINING GPR MODEL FOR {target} - {state_label}")
            print(f"Features: {final_variable_names}")
            print(f"Samples: {len(y_train)} train, {len(y_test)} test")
            print(f"{'='*60}")

            # Train on scaled y (normalize_y=False since we scale manually)
            gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42, normalize_y=False)
            gpr.fit(X_train, y_train_scaled)

            # Predictions (in scaled space)
            y_pred_train_scaled = gpr.predict(X_train)
            y_pred_test_scaled = gpr.predict(X_test)

            # Inverse transform predictions back to original scale
            y_pred_train = y_scaler.inverse_transform(y_pred_train_scaled.reshape(-1, 1)).ravel()
            y_pred_test = y_scaler.inverse_transform(y_pred_test_scaled.reshape(-1, 1)).ravel()

            # Metrics
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)

            # Create prediction plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # Train plot
            ax1.scatter(y_train, y_pred_train, alpha=0.6, edgecolors='k', linewidth=0.5)
            ax1.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()],
                     'r--', lw=2, label='Perfect prediction')
            ax1.set_xlabel('Actual Values', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Predicted Values', fontsize=12, fontweight='bold')
            ax1.set_title(f'Training Set (R² = {train_r2:.3f})', fontsize=14, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Test plot
            ax2.scatter(y_test, y_pred_test, alpha=0.6, edgecolors='k', linewidth=0.5, color='orange')
            ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                     'r--', lw=2, label='Perfect prediction')
            ax2.set_xlabel('Actual Values', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Predicted Values', fontsize=12, fontweight='bold')
            ax2.set_title(f'Test Set (R² = {test_r2:.3f})', fontsize=14, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            fig.suptitle(f'{target} - {state_label} - GPR Predictions vs Actual', fontsize=16, fontweight='bold')
            plt.tight_layout()

            plot_base64 = fig_to_base64(fig)

            # Store results for this state
            results[target][state_name] = {
                'model': gpr,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'plot': plot_base64,
                'features_used': final_variable_names,
                'features_removed': removed_features,
                'n_samples': len(y),
                'kernel': str(gpr.kernel_)
            }

            print(f"\n{'='*60}")
            print(f"COMPLETED {target} - {state_label}")
            print(f"{'='*60}")
            print(f"  Test R²: {test_r2:.4f}")
            print(f"  Test RMSE: {test_rmse:.4f}")
            print(f"  Test MAE: {test_mae:.4f}")
            print(f"  Features used: {len(final_variable_names)}")
            print(f"  Optimized kernel: {gpr.kernel_}")

    # Generate HTML report
    print(f"\n{'='*60}")
    print("GENERATING HTML REPORT")
    print(f"{'='*60}")
    create_html_report(results, output_file='gpr_report_dual_state_90m.html')

    # Save CSV summary
    print(f"\n{'='*60}")
    print("GENERATING CSV SUMMARY")
    print(f"{'='*60}")

    summary_data = []
    for target, state_results in results.items():
        # Low HR Activity
        if 'low_hr' in state_results:
            summary_data.append({
                'Target': target,
                'HR_Activity': 'Low',
                'Test_R2': state_results['low_hr']['test_r2'],
                'Test_RMSE': state_results['low_hr']['test_rmse'],
                'Test_MAE': state_results['low_hr']['test_mae'],
                'N_Features': len(state_results['low_hr']['features_used']),
                'Kernel': state_results['low_hr']['kernel']
            })

        # High HR Activity
        if 'high_hr' in state_results:
            summary_data.append({
                'Target': target,
                'HR_Activity': 'High',
                'Test_R2': state_results['high_hr']['test_r2'],
                'Test_RMSE': state_results['high_hr']['test_rmse'],
                'Test_MAE': state_results['high_hr']['test_mae'],
                'N_Features': len(state_results['high_hr']['features_used']),
                'Kernel': state_results['high_hr']['kernel']
            })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('gpr_summary_dual_state.csv', index=False)
    print("CSV summary saved to 'gpr_summary_dual_state.csv'")

    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print("\nGenerated:")
    print("  - gpr_report_dual_state_90m.html")
    print("  - gpr_summary_dual_state.csv")
    print(f"\nCoverage: {len(results)} clinical targets × 2 states")
