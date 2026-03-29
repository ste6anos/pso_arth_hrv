# Detection of Inflammatory Flares in Psoriatic Arthritis via HRV

> Diploma Thesis — Electrical and Computer Engineering  
> Aristotle University of Thessaloniki (AUTH), 2025  
> Supervisors: Leontios Hadjileontiadis, Georgios Apostolidis

📄 [Full thesis PDF](thesis_report.pdf)

---

## Table of Contents

1. [Abstract](#abstract)
2. [Dataset](#dataset)
3. [Methodology](#methodology)
4. [Results — Flare Detection](#results--flare-detection-via-hrv)
5. [Results — Disease Activity Scores](#results--predicting-disease-activity-scores-from-hrv)
6. [Limitations & Future Work](#limitations--future-work)

---

## Abstract

Psoriatic arthritis (PsA) is a chronic inflammatory autoimmune disease affecting the joints
and skin, causing pain, stiffness, and general malaise. During flare periods, symptoms can
be particularly severe, significantly impacting patients' quality of life. Early identification
or prediction of these flares could lead to more effective symptom management.

Heart rate variability (HRV) measures the variation in time between consecutive heartbeats
and reflects the activity of the autonomic nervous system. Low HRV values have been
associated with inflammatory conditions and physiological dysfunction, making it a promising
biomarker for monitoring chronic diseases. With modern wearable technology (smartwatches,
rings), continuous and non-invasive heart rate monitoring in daily life is now possible.

In this study, data from 111 PsA patients across Europe were collected via smartwatch.
The raw heart rate data underwent preprocessing to extract reliable HRV metrics.
Statistical analysis and machine learning techniques were then applied to investigate both
the discriminative and predictive ability of HRV features with respect to inflammation
flares and additional clinical variables.

---

## Dataset

The dataset used is the **PDPID (Psoriatic Disease Patient Identification) preliminary
dataset**, collected within the context of a European multi-center study.

### Cohort overview

| Attribute | Value |
|---|---|
| Patients | 111 adults with Psoriatic Arthritis (PsA) |
| Countries | Netherlands (26%), UK (29%), Greece (45%) |
| Collection period | September 2024 – April 2025 |
| HRV window used | First 14 days post-enrollment (T0) |

### Data sources

The dataset integrates three complementary data streams:

**1. Wearable physiological data**
Continuous beat-to-beat interval (BBI) recordings via smartwatch PPG
(Photoplethysmography), capturing inter-beat intervals in milliseconds throughout the day.

**2. Clinical assessment (physician)**
Recorded at enrollment visit (T0): joint counts (SJC/TJC), inflammatory markers (CRP),
composite disease activity scores (DAPSA, PASDAS, MDA), disease phenotype, and
physician-reported flare status (`DOC_FLARE`).

**3. Patient-reported outcomes (PROs)**
Questionnaire-based self-assessments including pain, fatigue, sleep quality, functional
status (HAQ), PsAID-12 score, and patient-reported flare status (`PAT_FLARE`).

### Key clinical variables

| Variable | Description | Missing |
|---|---|---|
| `DOC_FLARE` | Physician-reported inflammatory flare | 4% |
| `PAT_FLARE` | Patient-reported inflammatory flare | 27% |
| `CRP_mg_dL` | C-reactive protein (inflammation marker) | 23% |
| `DAPSA` | Disease Activity Index for Psoriatic Arthritis | 49% |
| `PASDAS` | Psoriatic Arthritis Disease Activity Score | 43% |
| `PSAID` | Patient Impact of Disease score | 26% |
| `HAQ` | Health Assessment Questionnaire | 25% |
| `BMI` | Body Mass Index | 5% |

### Patient demographics (Median [IQR])

- **Age:** 55 [45–61] years
- **Sex:** 51% male
- **Disease duration:** 8 [4–16] years
- **BMI:** 28 [25–34]
- **Psoriasis history:** 79%

> ⚠️ **Data availability:** The raw dataset is not publicly available due to patient privacy
> regulations. Only aggregated statistics and derived HRV metrics are used in the analyses
> presented in this repository.

---

## Methodology

The pipeline below summarises how raw smartwatch data was transformed into per-patient
HRV profiles ready for analysis. The key design choice is the **circadian split** (step 4):
separating rest and active periods ensures that differences observed between patients
reflect the disease rather than natural day/night HRV fluctuations.

![HRV Processing Pipeline](/hrv_pipeline.svg)

---

## Results — Flare Detection via HRV

The first research question was whether HRV can distinguish patients experiencing an
inflammatory flare from those who are not. Two binary outcomes were examined:

- **`DOC_FLARE`** — inflammatory flare as reported by the physician
- **`PAT_FLARE`** — inflammatory flare as self-reported by the patient

Each outcome was analysed separately for the rest/sleep and active/wake HRV profiles.

### Step 1 — Do HRV metrics differ between flare and no-flare patients?

Before building any predictive model, we tested whether HRV metrics were statistically
different between the two groups using Mann-Whitney U and independent t-tests.

| Clinical variable | HR state | Significant HRV metrics (p < 0.05) |
|---|---|---|
| `DOC_FLARE` | Rest/sleep | ULF, VLF, LF, TP |
| `DOC_FLARE` | Active/wake | RMSSD, SDNN |
| `PAT_FLARE` | Rest/sleep | ULF, VLF, LF, TP, LFHF, HFn |
| `PAT_FLARE` | Active/wake | RMSSD, SDNN, HTI |

> In all significant cases, **flare patients showed lower HRV values** — consistent with
> reduced parasympathetic (vagal) activity, which is theoretically linked to impaired
> anti-inflammatory regulation.

### Step 2 — Can we predict flares with a logistic regression model?

A logistic regression model was trained using HRV metrics combined with demographic
covariates (age, sex, BMI, CRP, smoking history). Four variants were tested per outcome,
combining:

- **SMOTE** (on/off) — synthetic oversampling to handle the imbalance between flare
  and no-flare cases
- **Threshold optimisation** (on/off) — tuning the classification cutoff beyond the
  default 0.5 to maximise F1-score on a validation set

#### `DOC_FLARE` — physician-reported flare

| Variant | HR state | Accuracy | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| No threshold, no SMOTE | Rest/sleep | 0.60 | 1.00 | 0.40 | **0.923** |
| No threshold, no SMOTE | Active/wake | 0.56 | 1.00 | 0.36 | **0.929** |
| **Threshold opt., no SMOTE** | **Rest/sleep** | **0.93** | **1.00** | **0.80** | **0.923** |
| Threshold opt., SMOTE | Rest/sleep | 0.80 | 1.00 | 0.57 | 0.962 |

#### `PAT_FLARE` — patient-reported flare

| Variant | HR state | Accuracy | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| No threshold, no SMOTE | Active/wake | 0.85 | 1.00 | 0.50 | **0.917** |
| No threshold, SMOTE | Active/wake | 0.85 | 1.00 | 0.50 | **0.917** |
| Threshold opt., no SMOTE | Rest/sleep | 0.83 | 1.00 | 0.50 | 0.818 |
| **Threshold opt., SMOTE** | **Active/wake** | **0.85** | **1.00** | **0.67** | **0.864** |

### Key takeaways

- **HRV carries a meaningful signal for flare detection** — ROC-AUC consistently above
  0.85 across both flare definitions.
- The **rest/sleep profile** was most informative for `DOC_FLARE`, with low-frequency
  spectral metrics (ULF, VLF, LF) driving the separation.
- The **active/wake profile** was most informative for `PAT_FLARE`, with time-domain
  metrics (RMSSD, SDNN) being the primary discriminators.
- **Recall = 1.00** across all best models — no flare case was missed, which is the
  clinically critical direction.
- Results should be interpreted as **exploratory** given the small sample size (n < 80
  after missing data exclusion).

---

## Results — Predicting Disease Activity Scores from HRV

The second research question was whether HRV can go beyond binary detection and
**quantitatively predict how severe** a patient's disease is, as measured by continuous
clinical scores.

| Score | What it measures |
|---|---|
| **DAPSA** | Joint inflammation and patient-reported symptoms (range 0–164) |
| **PASDAS** | Composite disease activity across joints, skin and quality of life |
| **PSAID** | Patient-perceived impact of PsA on quality of life (range 0–10) |
| **HAQ** | Functional disability in daily activities (range 0–3) |
| **BMI** | Body mass index — included as a metabolic reference variable |
| **CRP** | C-reactive protein — objective blood marker of inflammation |

Each score was examined against all 13 HRV metrics, separately for the rest/sleep and
active/wake profiles.

### Methods

Two machine learning approaches were applied and compared:

**Symbolic Regression (PySR)**
An evolutionary algorithm that searches for explicit mathematical formulas linking HRV
metrics to the target score (e.g. `BMI ≈ 38.20 + HTI / (−0.27 × HFn)`). The result is
a human-readable equation rather than a black box — but it is prone to overfitting on
small samples.

**Random Forest Regression**
An ensemble of 200 decision trees, each trained on a random subset of the data. More
robust to noise than symbolic regression, and provides feature importance scores showing
which HRV metrics contribute most. Both methods were evaluated on a held-out test set
(20%) using R² and RMSE.

### Results

| Target | Method | HR state | Train R² | Test R² |
|---|---|---|---|---|
| **DAPSA** | Symbolic Reg. | Rest/sleep | 0.06 | -1.42 |
| **DAPSA** | Symbolic Reg. | Active/wake | 0.12 | -0.59 |
| **DAPSA** | Random Forest | Rest/sleep | -0.12 | -1.50 |
| **DAPSA** | Random Forest | Active/wake | -0.06 | -0.88 |
| **PASDAS** | Symbolic Reg. | Rest/sleep | 0.03 | -0.96 |
| **PASDAS** | Symbolic Reg. | Active/wake | 0.07 | -0.73 |
| **PASDAS** | Random Forest | Rest/sleep | -0.13 | -0.59 |
| **PASDAS** | Random Forest | Active/wake | 0.03 | -0.70 |
| **PSAID** | Symbolic Reg. | Rest/sleep | 0.04 | -0.36 |
| **PSAID** | Symbolic Reg. | Active/wake | 0.02 | -0.09 |
| **PSAID** | Random Forest | Rest/sleep | 0.004 | -0.70 |
| **PSAID** | Random Forest | Active/wake | 0.02 | -0.27 |
| **HAQ** | Symbolic Reg. | Rest/sleep | 0.02 | -0.16 |
| **HAQ** | Symbolic Reg. | Active/wake | 0.06 | **+0.04** |
| **HAQ** | Random Forest | Rest/sleep | 0.01 | -0.28 |
| **HAQ** | Random Forest | Active/wake | -0.02 | -0.12 |
| **BMI** | Symbolic Reg. | Rest/sleep | 0.11 | **+0.25** |
| **BMI** | Symbolic Reg. | Active/wake | 0.03 | -0.27 |
| **BMI** | Random Forest | Rest/sleep | 0.10 | -0.21 |
| **BMI** | Random Forest | Active/wake | 0.13 | -0.31 |
| **CRP** | Symbolic Reg. | Rest/sleep | 0.01 | -0.41 |
| **CRP** | Symbolic Reg. | Active/wake | **0.94** | -0.09 |
| **CRP** | Random Forest | Rest/sleep | 0.01 | -0.50 |
| **CRP** | Random Forest | Active/wake | 0.11 | -0.20 |

### Commentary

The models failed to produce meaningful predictions for any of the scores. Negative test
R² values indicate that the models perform **worse than simply predicting the mean** — a
sign that HRV metrics, as computed here, do not carry sufficient signal to quantify disease
severity on a continuous scale.

This is consistent with the correlation analysis, which found no statistically significant
linear relationship between any HRV metric and DAPSA, PASDAS, PSAID, or HAQ
(all p > 0.05 across both HR states).

Two notable exceptions worth flagging:

- **CRP — Symbolic Reg., active/wake (Train R² = 0.94):** a case of severe overfitting.
  The algorithm found a formula that fits the training data almost perfectly but completely
  fails to generalise (Test R² = −0.09). A known risk of symbolic regression on small
  samples.
- **BMI — Symbolic Reg., rest/sleep (Test R² = +0.25):** the only model that achieved
  a positive test R², suggesting a weak but non-trivial link between resting HRV and body
  composition — consistent with known associations between autonomic tone and metabolic
  status.

Two likely explanations for the overall poor performance:

- **Small sample size.** Effective sample sizes dropped to 49–57 patients for DAPSA and
  PASDAS after excluding missing values — far below what regression models typically
  require to generalise.
- **Signal mismatch.** Composite scores like DAPSA and PASDAS aggregate joint counts,
  patient-reported pain, and lab values. HRV may reflect systemic inflammatory burden
  but not the full clinical picture these scores capture.

> HRV shows promise as a **binary alarm signal** (flare vs. no flare) rather than a
> continuous gauge of disease severity — at least at this sample size and time resolution.

---

## Limitations & Future Work

### Limitations

- **Small sample size.** With 111 patients total and up to 49% missing values on some
  clinical scores, effective sample sizes were often below 60 — limiting the power of
  both statistical tests and machine learning models.
- **No sleep ground truth.** The rest/wake split was derived from HR thresholds rather
  than validated sleep detection. A dedicated sleep-tracking algorithm or actigraphy
  data would improve the circadian separation.
- **Single wearable device.** All data was collected from one smartwatch model. HRV
  metrics derived from PPG signals are less accurate than ECG-based measurements,
  and results may not generalise across devices.
- **Flare imbalance.** Only 19% of patients had a physician-reported flare at T0,
  creating a class imbalance that required SMOTE augmentation and limits confidence
  in the minority class predictions.

### Future directions

- **Larger cohort.** The PDPID study is ongoing — a larger sample would allow more
  robust regression models and potentially reveal HRV–disease score associations not
  detectable here.
- **Longer HRV windows.** Low-frequency HRV bands (ULF, VLF) require recordings
  of several hours to be reliably estimated. Extending the analysis window beyond
  90 minutes could strengthen the spectral signal.
- **Validated sleep staging.** Replacing the HR-threshold split with a validated
  sleep detection algorithm would provide a cleaner circadian separation and more
  interpretable rest/wake HRV profiles.
- **Multimodal integration.** Combining HRV with other wearable signals (activity,
  skin temperature, SpO₂) and patient-reported daily symptom logs could substantially
  improve predictive models.
