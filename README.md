# Air Quality Anomaly Detection in Industrial Regions

This project implements an automated machine learning pipeline to predict and detect extreme nitrogen dioxide ($NO_2$) pollution events in major industrial cities of Ukraine (Zaporizhzhia, Dnipro, and Kryvyi Rih). By integrating satellite imagery with high-precision meteorological data, the system identifies the specific weather conditions that lead to critical smog accumulation.

---

## Project Objective

Continuous industrial emissions combined with adverse meteorological conditions often create a "trap" for pollutants. The primary goal of this research is to move beyond simple observation and build a predictive system that can proactively identify days where $NO_2$ surface concentrations will exceed the WHO safety limit of 25 $μg/m^3$.

---

## Data Sources

The dataset is aggregated via the Google Earth Engine (GEE) API, synchronizing spatial and temporal data from 2020 to 2025:

* **Copernicus Sentinel-5P (TROPOMI):** Provides daily satellite measurements of atmospheric gas column densities (including $NO_2$, $SO_2$, $CO$, and Aerosol Index).
* **ECMWF ERA5-Land:** High-resolution surface meteorological data (temperature, dewpoint, wind components).
* **ECMWF ERA5:** Hourly reanalysis data providing the Planetary Boundary Layer Height (PBLH), a critical metric for calculating the physical volume available for emission dispersion.

---

## Pipeline Architecture

1. **Data Integration & Filtering:** Fetching daily satellite and weather data for 15km buffers around the target cities. Cloudy pixels (cloud fraction > 0.3) are masked to ensure chemical measurement validity.
2. **Physical Transformation:** Converting raw satellite column densities into surface-level $NO_2$ concentrations ($μg/m^3$) using the Planetary Boundary Layer Height. 
3. **Target Variable Creation:** Generating a binary classification target (1 = Anomaly, 0 = Normal) based on the WHO 25 $μg/m^3$ threshold.
4. **Feature Engineering:** Creating temporal lags (e.g., $NO_2$ levels 1, 2, and 7 days prior), rolling averages (3-day memory), and calendar features (month, day of week, weekend indicator) to capture accumulation trends and industrial cycles.
5. **Data Scaling & Leakage Prevention:** Removing direct current-day chemical measurements from the feature set to prevent data leakage and standardizing features using `StandardScaler`.

---

## Machine Learning Models

The project frames smog prediction as a highly imbalanced binary classification problem. Four different algorithms were evaluated to find the optimal balance between detecting real threats (Recall) and minimizing false alarms (Precision).

| Model | Precision | Recall | F1-Score | PR-AUC |
| :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.4286 | 0.9310 | 0.5870 | 0.7987 |
| **Random Forest** | 0.8889 | 0.5517 | 0.6809 | 0.8074 |
| **SVM (RBF Kernel)** | 0.4783 | 0.7586 | 0.5867 | 0.7296 |
| **XGBoost** | 0.6875 | 0.7586 | 0.7213 | 0.7529 |

### Key Findings
* **XGBoost** was selected as the final production model. It successfully handles the severe class imbalance (using `scale_pos_weight`) and captures complex, non-linear meteorological patterns, delivering the highest overall F1-Score and a stable Precision-Recall balance.
* **Feature Importance:** The model explicitly confirmed the physical nature of the smog problem. The **Boundary Layer Height** accounts for over 60% of the model's predictive power, followed by the 3-day rolling mean of $NO_2$ and wind velocity components.

---
