# Detection of Inflammatory Flares in Psoriatic Arthritis through Heart Rate Variability (HRV)

**Diploma Thesis** **Course:** Electrical and Computer Engineering Undergraduate Program  
**Institution:** Aristotle University of Thessaloniki (AUTH)  
**Academic Year:** [2025]

## Documentation
For a detailed analysis of the methodology and clinical results, you can:
* [Check the full thesis review pdf](thesis_report.pdf)

## Abstract
This thesis investigates the ability to identify inflammation flares and other clinical va-
riables through heart rate data in patients with psoriatic arthritis.
Psoriatic arthritis is a chronic inflammatory autoimmune disease that affects the joints
andskin, causingpain, stiffness, andgeneralmalaise. Duringflareperiods, symptomscan
beparticularlysevere, significantlyimpactingpatients’qualityoflife. Earlyidentification
or prediction of these flares could lead to more effective symptom management.

Heart rate variability (HRV) is a measure of the variation in time between consecutive
heartbeats and reflects the activity of the autonomic nervous system. Low HRV values
have been associated with inflammatory conditions and physiological dysfunction, ma-
king it a promising biomarker for monitoring chronic diseases. With modern wearable
technology (smartwatches, rings), continuous and non-invasive heart rate monitoring in
daily life is now possible.

In this study, data from 111 patients across Europe were collected via smartwatch. The
raw heart rate data underwent preprocessing to extract reliable HRV metrics. Subseque-
ntly, statistical analysis and machine learning techniques were applied to investigate both
the discriminative and predictive ability of HRV features with respect to inflammation
flares, as well as additional clinical variables of the disease.


## Dataset – PDPID Preliminary Cohort

The dataset used in this study is the **PDPID (Psoriatic Disease Patient Identification) preliminary dataset**, collected within the context of a European multi-center study.

### Cohort Overview

| Attribute | Value |
|-----------|-------|
| Patients | 111 adults with Psoriatic Arthritis (PsA) |
| Countries | Netherlands (26%), UK (29%), Greece (45%) |
| Collection Period | September 2024 – April 2025 |
| HRV Window Used | First 14 days post-enrollment (T0) |

### Data Sources

The dataset integrates three complementary data streams:

**1. Wearable Physiological Data**  
Continuous beat-to-beat interval (BBI) recordings via smartwatch PPG (Photoplethysmography), capturing inter-beat intervals in milliseconds throughout the day.

**2. Clinical Assessment (Physician)**  
Recorded at enrollment visit (T0): joint counts (SJC/TJC), inflammatory markers (CRP), composite disease activity scores (DAPSA, PASDAS, MDA), disease phenotype, and physician-reported flare status (DOC_FLARE).

**3. Patient-Reported Outcomes (PROs)**  
Questionnaire-based self-assessments including: pain, fatigue, sleep quality, functional status (HAQ), PsAID-12 score, and patient-reported flare status (PAT_FLARE).

### Key Clinical Variables

| Variable | Description | Missing |
|----------|-------------|---------|
| `DOC_FLARE` | Physician-reported inflammatory flare | 4% |
| `PAT_FLARE` | Patient-reported inflammatory flare | 27% |
| `CRP_mg_dL` | C-reactive protein (inflammation marker) | 23% |
| `DAPSA` | Disease Activity Index for PSoriatic Arthritis | 49% |
| `PASDAS` | Psoriatic Arthritis Disease Activity Score | 43% |
| `PsAID-12` | Patient Impact of Disease score | 26% |
| `HAQ` | Health Assessment Questionnaire | 25% |
| `BMI` | Body Mass Index | 5% |

### Patient Demographics (Median [IQR])

- **Age:** 55 [45–61] years  
- **Sex:** 51% male  
- **Disease duration:** 8 [4–16] years  
- **BMI:** 28 [25–34]  
- **Psoriasis history:** 79%

> ⚠️ **Data Availability:** The raw dataset is not publicly available due to patient privacy regulations. Only aggregated statistics and derived HRV metrics are used in the analyses presented in this repository.
