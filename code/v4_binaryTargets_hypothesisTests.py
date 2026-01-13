import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, ttest_ind, shapiro
from datetime import datetime


def smart_format(value):
    """
    Smart formatting that shows appropriate decimal places based on value magnitude.
    For very small values, shows more decimals to capture the variation.
    """
    if pd.isna(value) or value == 0:
        return "0.00"

    abs_val = abs(value)

    # For values >= 1, use 2-3 decimals
    if abs_val >= 1:
        return f"{value:.2f}"
    # For values 0.1 to 1, use 3 decimals
    elif abs_val >= 0.1:
        return f"{value:.3f}"
    # For values 0.01 to 0.1, use 4 decimals
    elif abs_val >= 0.01:
        return f"{value:.4f}"
    # For very small values < 0.01, use 5-6 decimals
    elif abs_val >= 0.0001:
        return f"{value:.5f}"
    else:
        # Use scientific notation for extremely small values
        return f"{value:.2e}"


def hypothesis_test_binary_target(group1_df, group2_df, metrics):
    """
    Perform hypothesis testing (t-test or Mann-Whitney U) for binary target groups.

    Parameters:
    - group1_df: DataFrame for group 1 (e.g., Male, Yes, etc.)
    - group2_df: DataFrame for group 2 (e.g., Female, No, etc.)
    - metrics: List of HRV metric column names

    Returns:
    - DataFrame with p-values and test methods for each metric
    """
    results = []

    for metric in metrics:
        if metric not in group1_df.columns or metric not in group2_df.columns:
            continue

        data1 = group1_df[metric].dropna()
        data2 = group2_df[metric].dropna()

        if len(data1) < 3 or len(data2) < 3:
            continue

        # Check normality
        _, p1 = shapiro(data1) if len(data1) >= 3 else (None, 0)
        _, p2 = shapiro(data2) if len(data2) >= 3 else (None, 0)

        # Choose test based on normality
        if p1 > 0.05 and p2 > 0.05:
            # Both normal → t-test
            stat, p_value = ttest_ind(data1, data2)
            method = 't-test'
        else:
            # At least one non-normal → Mann-Whitney U
            stat, p_value = mannwhitneyu(data1, data2, alternative='two-sided')
            method = 'Mann-Whitney U'

        # Calculate descriptive statistics for both groups
        mean1, std1 = data1.mean(), data1.std()
        mean2, std2 = data2.mean(), data2.std()
        median1, q1_1, q3_1 = data1.median(), data1.quantile(0.25), data1.quantile(0.75)
        median2, q1_2, q3_2 = data2.median(), data2.quantile(0.25), data2.quantile(0.75)

        results.append({
            'metric': metric,
            'p_value': p_value,
            'method': method,
            'n_group1': len(data1),
            'n_group2': len(data2),
            'mean_group1': mean1,
            'std_group1': std1,
            'median_group1': median1,
            'q1_group1': q1_1,
            'q3_group1': q3_1,
            'mean_group2': mean2,
            'std_group2': std2,
            'median_group2': median2,
            'q1_group2': q1_2,
            'q3_group2': q3_2
        })

    return pd.DataFrame(results)


def create_html_report(results_dict, output_file='binary_targets_hypothesis_tests.html'):
    """Generate minimal HTML report for binary target hypothesis tests."""

    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Binary Targets Hypothesis Tests</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
            line-height: 1.6;
            color: #2c3e50;
            background: #f8f9fa;
            padding: 0;
        }}

        .container {{
            max-width: 1800px;
            margin: 0 auto;
            background: white;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}

        .header {{
            background: white;
            border-bottom: 2px solid #e9ecef;
            padding: 40px;
            text-align: center;
        }}

        .header h1 {{
            font-size: 2em;
            margin-bottom: 8px;
            font-weight: 600;
            color: #2c3e50;
        }}

        .header p {{
            font-size: 0.95em;
            color: #6c757d;
        }}

        .content {{
            padding: 40px;
        }}

        .legend {{
            background: #f6f8fa;
            padding: 20px;
            border: 1px solid #e1e4e8;
            border-radius: 4px;
            margin-bottom: 32px;
        }}

        .legend h3 {{
            font-size: 0.875em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-weight: 600;
            margin-bottom: 12px;
            color: #24292e;
        }}

        .legend-items {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 8px;
        }}

        .legend-item {{
            padding: 8px 12px;
            border: 1px solid #e1e4e8;
            border-radius: 3px;
            font-size: 0.875em;
        }}

        .target-section {{
            margin-bottom: 60px;
        }}

        .target-section h2 {{
            color: #2c3e50;
            margin-bottom: 20px;
            font-size: 1.5em;
            font-weight: 600;
            padding-bottom: 10px;
            border-bottom: 1px solid #dee2e6;
        }}

        .comparison-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.875em;
            background: white;
            border: 1px solid #e1e4e8;
        }}

        th {{
            background: #f6f8fa;
            color: #24292e;
            padding: 10px 12px;
            text-align: left;
            font-weight: 600;
            border-bottom: 2px solid #e1e4e8;
            font-size: 0.8em;
            text-transform: uppercase;
            letter-spacing: 0.3px;
        }}

        td {{
            padding: 8px 12px;
            border-bottom: 1px solid #e1e4e8;
        }}

        tr:hover {{
            background: #f6f8fa;
        }}

        .sig {{
            background: #d4edda;
            font-weight: 600;
        }}

        .not-sig {{
            background: #ffffff;
        }}

        .panel {{
            background: #fafbfc;
            border: 1px solid #e1e4e8;
            border-radius: 4px;
            padding: 16px;
        }}

        .panel h3 {{
            font-size: 1em;
            font-weight: 600;
            margin-bottom: 12px;
            color: #24292e;
        }}

        .stat-badge {{
            display: inline-block;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 0.75em;
            font-weight: 600;
            margin-left: 8px;
        }}

        .badge-sig {{
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }}

        .badge-not-sig {{
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
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
            .comparison-grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Binary Targets Hypothesis Tests</h1>
            <p>HRV Metrics Discrimination Analysis - Low & High HR Activity States</p>
            <p style="font-size: 0.85em; margin-top: 8px; color: #6c757d;">
                Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            </p>
        </div>

        <div class="content">
            <div class="legend">
                <h3>Interpretation Guide</h3>
                <div class="legend-items">
                    <div class="legend-item sig">
                        <strong>p &lt; 0.05:</strong> Significant difference (green highlight)
                    </div>
                    <div class="legend-item not-sig">
                        <strong>p ≥ 0.05:</strong> No significant difference
                    </div>
                    <div class="legend-item" style="background: #fff3cd;">
                        <strong>Method:</strong> t-test (normal) or Mann-Whitney U (non-normal)
                    </div>
                    <div class="legend-item" style="background: #e7f3ff;">
                        <strong>Values:</strong> Mean ± SD (t-test) or Median [IQR] (Mann-Whitney U)
                    </div>
                </div>
            </div>
"""

    # Add each target section
    for target, data in results_dict.items():
        target_info = data['info']
        low_hr_results = data['low_hr']
        high_hr_results = data['high_hr']

        html += f"""
            <div class="target-section">
                <h2>{target}</h2>
                <p style="margin-bottom: 16px; color: #586069; font-size: 0.9em;">
                    {target_info['description']}
                </p>

                <div class="comparison-grid">
                    <div class="panel">
                        <h3>Low HR Activity</h3>
"""

        if not low_hr_results.empty:
            html += f"""
                        <table>
                            <thead>
                                <tr>
                                    <th>HRV Metric</th>
                                    <th>p-value</th>
                                    <th>Method</th>
                                    <th>{target_info['group1_label']} (n)</th>
                                    <th>{target_info['group1_label']} Values</th>
                                    <th>{target_info['group2_label']} (n)</th>
                                    <th>{target_info['group2_label']} Values</th>
                                </tr>
                            </thead>
                            <tbody>
"""
            for _, row in low_hr_results.iterrows():
                sig_class = 'sig' if row['p_value'] < 0.05 else 'not-sig'
                metric_short = row['metric'].replace('HRV_', '')
                p_display = f"{row['p_value']:.4f}" if row['p_value'] >= 0.001 else "<0.001"

                # Format group values based on test method
                if row['method'] == 't-test':
                    g1_values = f"{smart_format(row['mean_group1'])} ± {smart_format(row['std_group1'])}"
                    g2_values = f"{smart_format(row['mean_group2'])} ± {smart_format(row['std_group2'])}"
                else:
                    iqr1 = row['q3_group1'] - row['q1_group1']
                    iqr2 = row['q3_group2'] - row['q1_group2']
                    g1_values = f"{smart_format(row['median_group1'])} [{smart_format(iqr1)}]"
                    g2_values = f"{smart_format(row['median_group2'])} [{smart_format(iqr2)}]"

                html += f"""
                                <tr class="{sig_class}">
                                    <td><strong>{metric_short}</strong></td>
                                    <td>{p_display}</td>
                                    <td>{row['method']}</td>
                                    <td>{int(row['n_group1'])}</td>
                                    <td>{g1_values}</td>
                                    <td>{int(row['n_group2'])}</td>
                                    <td>{g2_values}</td>
                                </tr>
"""
            html += """
                            </tbody>
                        </table>
"""
        else:
            html += '<p style="color: #6c757d; text-align: center; padding: 20px;">No data available</p>'

        html += """
                    </div>
                    <div class="panel">
                        <h3>High HR Activity </h3>
"""

        if not high_hr_results.empty:
            html += f"""
                        <table>
                            <thead>
                                <tr>
                                    <th>HRV Metric</th>
                                    <th>p-value</th>
                                    <th>Method</th>
                                    <th>{target_info['group1_label']} (n)</th>
                                    <th>{target_info['group1_label']} Values</th>
                                    <th>{target_info['group2_label']} (n)</th>
                                    <th>{target_info['group2_label']} Values</th>
                                </tr>
                            </thead>
                            <tbody>
"""
            for _, row in high_hr_results.iterrows():
                sig_class = 'sig' if row['p_value'] < 0.05 else 'not-sig'
                metric_short = row['metric'].replace('HRV_', '')
                p_display = f"{row['p_value']:.4f}" if row['p_value'] >= 0.001 else "<0.001"

                # Format group values based on test method
                if row['method'] == 't-test':
                    g1_values = f"{smart_format(row['mean_group1'])} ± {smart_format(row['std_group1'])}"
                    g2_values = f"{smart_format(row['mean_group2'])} ± {smart_format(row['std_group2'])}"
                else:
                    iqr1 = row['q3_group1'] - row['q1_group1']
                    iqr2 = row['q3_group2'] - row['q1_group2']
                    g1_values = f"{smart_format(row['median_group1'])} [{smart_format(iqr1)}]"
                    g2_values = f"{smart_format(row['median_group2'])} [{smart_format(iqr2)}]"

                html += f"""
                                <tr class="{sig_class}">
                                    <td><strong>{metric_short}</strong></td>
                                    <td>{p_display}</td>
                                    <td>{row['method']}</td>
                                    <td>{int(row['n_group1'])}</td>
                                    <td>{g1_values}</td>
                                    <td>{int(row['n_group2'])}</td>
                                    <td>{g2_values}</td>
                                </tr>
"""
            html += """
                            </tbody>
                        </table>
"""
        else:
            html += '<p style="color: #6c757d; text-align: center; padding: 20px;">No data available</p>'

        html += """
                    </div>
                </div>
            </div>
"""

    html += """
        </div>

        <div class="footer">
            <p>Binary Targets Hypothesis Testing | HRV Metrics Analysis</p>
            <p style="margin-top: 8px; font-size: 0.85em;">
                Low HR Activity | High HR Activity
            </p>
        </div>
    </div>
</body>
</html>
"""

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"\nHTML report saved to '{output_file}'")


# ===========================================================================
# MAIN EXECUTION
# ===========================================================================

if __name__ == "__main__":
    print("="*80)
    print("BINARY TARGETS HYPOTHESIS TESTING")
    print("="*80)

    # Load data
    clinical_data_df = pd.read_csv(r"C:\Users\spbtu\Documents\dataset_psath\bbi_metadata\clinical_d32_T0_v2.csv")
    df_sl = pd.read_csv(r"C:\Users\spbtu\Documents\hrvThesis\window90m_hrv_mean_measurements_sl.csv", index_col=0)
    df_aw = pd.read_csv(r"C:\Users\spbtu\Documents\hrvThesis\window90m_hrv_mean_measurements_aw.csv", index_col=0)

    # Define HRV metrics
    hrv_metrics = [
        "HRV_RMSSD", "HRV_SDNN", "HRV_HTI",
        "HRV_ULF", "HRV_VLF", "HRV_LF", "HRV_HF", "HRV_VHF",
        "HRV_TP", "HRV_LFHF", "HRV_LFn", "HRV_HFn", "HRV_LnHF"
    ]

    # Define binary targets
    binary_targets = {
        "DEMOGR_SEX": {
            "description": "Gender comparison: Male vs Female",
            "group1_label": "Male",
            "group2_label": "Female",
            "group1_values": ["Male"],
            "group2_values": ["Female"]
        },
        "PAT_SMOKE_PAST_H": {
            "description": "Past smoking history: Yes vs No",
            "group1_label": "Yes",
            "group2_label": "No",
            "group1_values": ["Yes", 1],
            "group2_values": ["No", 0]
        },
        "DOC_FLARE": {
            "description": "Doctor-reported flare: Yes vs No",
            "group1_label": "Yes (Flare)",
            "group2_label": "No (No Flare)",
            "group1_values": ["Yes", 1],
            "group2_values": ["No", 0]
        },
        "PAT_FLARE": {
            "description": "Patient-reported flare: Yes vs No",
            "group1_label": "Yes (Flare)",
            "group2_label": "No (No Flare)",
            "group1_values": ["Yes", 1],
            "group2_values": ["No", 0]
        },
        "DOC_DIS_ACT": {
            "description": "Doctor-assessed disease activity: Active vs Inactive",
            "group1_label": "Yes (Active)",
            "group2_label": "No (Inactive)",
            "group1_values": ["Yes", 1],
            "group2_values": ["No", 0]
        }
    }

    results_dict = {}

    # Process each target
    for target, target_config in binary_targets.items():
        print(f"\nProcessing {target}...")
        print(f"  {target_config['description']}")

        # Get patient IDs for each group
        group1_ids = clinical_data_df[
            clinical_data_df[target].isin(target_config['group1_values'])
        ]['patientid']

        group2_ids = clinical_data_df[
            clinical_data_df[target].isin(target_config['group2_values'])
        ]['patientid']

        print(f"  Group 1 ({target_config['group1_label']}): {len(group1_ids)} patients")
        print(f"  Group 2 ({target_config['group2_label']}): {len(group2_ids)} patients")

        # Low HR Activity 
        print(f"  - Testing Low HR Activity ...")
        sl_group1 = df_sl[df_sl['patientid'].isin(group1_ids)]
        sl_group2 = df_sl[df_sl['patientid'].isin(group2_ids)]

        low_hr_results = hypothesis_test_binary_target(sl_group1, sl_group2, hrv_metrics)
        sig_low = low_hr_results[low_hr_results['p_value'] < 0.05]
        n_sig_low = len(sig_low)
        print(f"    Significant metrics: {n_sig_low}/{len(low_hr_results)}")

        if n_sig_low > 0:
            print(f"\n    >> Significant differences in Low HR Activity (p<0.05):")
            for _, row in sig_low.iterrows():
                metric_short = row['metric'].replace('HRV_', '')
                if row['method'] == 't-test':
                    g1_str = f"{smart_format(row['mean_group1'])}±{smart_format(row['std_group1'])}"
                    g2_str = f"{smart_format(row['mean_group2'])}±{smart_format(row['std_group2'])}"
                else:
                    iqr1 = row['q3_group1'] - row['q1_group1']
                    iqr2 = row['q3_group2'] - row['q1_group2']
                    g1_str = f"{smart_format(row['median_group1'])}[IQR:{smart_format(iqr1)}]"
                    g2_str = f"{smart_format(row['median_group2'])}[IQR:{smart_format(iqr2)}]"
                print(f"      • {metric_short:12s} | p={row['p_value']:.4f} | {target_config['group1_label']}: {g1_str} | {target_config['group2_label']}: {g2_str}")

        # High HR Activity 
        print(f"  - Testing High HR Activity ...")
        aw_group1 = df_aw[df_aw['patientid'].isin(group1_ids)]
        aw_group2 = df_aw[df_aw['patientid'].isin(group2_ids)]

        high_hr_results = hypothesis_test_binary_target(aw_group1, aw_group2, hrv_metrics)
        sig_high = high_hr_results[high_hr_results['p_value'] < 0.05]
        n_sig_high = len(sig_high)
        print(f"    Significant metrics: {n_sig_high}/{len(high_hr_results)}")

        if n_sig_high > 0:
            print(f"\n    >> Significant differences in High HR Activity (p<0.05):")
            for _, row in sig_high.iterrows():
                metric_short = row['metric'].replace('HRV_', '')
                if row['method'] == 't-test':
                    g1_str = f"{smart_format(row['mean_group1'])}±{smart_format(row['std_group1'])}"
                    g2_str = f"{smart_format(row['mean_group2'])}±{smart_format(row['std_group2'])}"
                else:
                    iqr1 = row['q3_group1'] - row['q1_group1']
                    iqr2 = row['q3_group2'] - row['q1_group2']
                    g1_str = f"{smart_format(row['median_group1'])}[IQR:{smart_format(iqr1)}]"
                    g2_str = f"{smart_format(row['median_group2'])}[IQR:{smart_format(iqr2)}]"
                print(f"      • {metric_short:12s} | p={row['p_value']:.4f} | {target_config['group1_label']}: {g1_str} | {target_config['group2_label']}: {g2_str}")

        # Store results
        results_dict[target] = {
            'info': target_config,
            'low_hr': low_hr_results,
            'high_hr': high_hr_results
        }

    # Generate HTML report
    print("\n" + "="*80)
    print("Generating HTML report...")
    print("="*80)
    create_html_report(results_dict, output_file='binary_targets_hypothesis_tests_w90m.html')

    # Save summary CSV
    print("\nGenerating summary CSV...")
    summary_data = []
    for target, data in results_dict.items():
        target_info = data['info']
        for state, state_label in [('low_hr', 'Low HR'), ('high_hr', 'High HR')]:
            results_df = data[state]
            for _, row in results_df.iterrows():
                # Format values based on test method
                if row['method'] == 't-test':
                    g1_values = f"{smart_format(row['mean_group1'])}±{smart_format(row['std_group1'])}"
                    g2_values = f"{smart_format(row['mean_group2'])}±{smart_format(row['std_group2'])}"
                else:
                    iqr1 = row['q3_group1'] - row['q1_group1']
                    iqr2 = row['q3_group2'] - row['q1_group2']
                    g1_values = f"{smart_format(row['median_group1'])}[IQR:{smart_format(iqr1)}]"
                    g2_values = f"{smart_format(row['median_group2'])}[IQR:{smart_format(iqr2)}]"

                summary_data.append({
                    'Target': target,
                    'State': state_label,
                    'Metric': row['metric'],
                    'p_value': row['p_value'],
                    'Significant': 'Yes' if row['p_value'] < 0.05 else 'No',
                    'Method': row['method'],
                    'Group1_Label': target_info['group1_label'],
                    'n_group1': row['n_group1'],
                    'Group1_Values': g1_values,
                    'Group2_Label': target_info['group2_label'],
                    'n_group2': row['n_group2'],
                    'Group2_Values': g2_values
                })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('binary_targets_hypothesis_tests_summary.csv', index=False)
    print("Summary CSV saved to 'binary_targets_hypothesis_tests_summary.csv'")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nGenerated:")
    print("  - binary_targets_hypothesis_tests_w90m.html")
    print("  - binary_targets_hypothesis_tests_summary.csv")
    print(f"\nCoverage: {len(binary_targets)} binary targets × 2 states")
