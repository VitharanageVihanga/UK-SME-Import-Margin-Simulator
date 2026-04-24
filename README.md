# 🇬UK SME Import Margin Simulator

A **data-driven Streamlit dashboard** that helps UK small and medium enterprises (SMEs) model import profitability, assess financial risk, and forecast margins under varying economic conditions.

Built as a **Final Year Project** using real trade data from HMRC and ONS.

---

##  Overview

This project provides an interactive analytics platform for SMEs to:

* Simulate import margins under different cost conditions
* Evaluate financial risk using statistical and machine learning techniques
* Forecast exchange rates and profitability
* Analyze historical trade trends and volatility regimes

---

## Features

* **Margin Simulation**
  Calculate landed cost and profit margin with adjustable FX rate, shipping, tariff, and overhead inputs

* **Sensitivity Analysis**
  11×11 grid showing how margins react to FX and shipping shocks

* **Risk Classification**
  Automatic HIGH / MODERATE / LOW risk labeling with confidence adjustments

* **Historical Trend Analysis**
  Volatility detection, moving averages, and import volume trends (2018–2024)

* **FX Forecasting**
  ARIMA-based 30/90-day forecasts with confidence intervals

* **Advanced Risk Metrics**
  Value-at-Risk (VaR), beta sensitivity, and margin decomposition

* **Stress Testing**
  Seven scenarios including FX shock, supply disruption, and “Perfect Storm”

* **Anomaly Detection**
  Isolation Forest model to detect unusual patterns

* **Market Intelligence**
  Commodity-level trade analysis using HMRC + ONS datasets

* **Backtesting**
  Walk-forward ARIMA validation with MAE, RMSE, MAPE, and directional accuracy

---

##  Tech Stack

| Layer           | Technology                   |
| --------------- | ---------------------------- |
| Dashboard       | Streamlit 1.56, Plotly 6.7   |
| Data Processing | pandas 3.0, NumPy 2.4        |
| Forecasting     | statsmodels 0.14 (ARIMA)     |
| Risk / ML       | scikit-learn 1.8, scipy 1.17 |
| Testing         | pytest 9.0, pytest-cov 7.1   |
| Runtime         | Python 3.11                  |

---

## Project Structure

```
├── app.py
├── requirements.txt
├── scripts/
├── tests/
├── data/                # (NOT included in repo – see setup below)
├── QA_Report.html
└── WORKFLOW.md
```

---

## Important Note on Data

Due to GitHub file size limitations, the dataset is **not included** in this repository.

You must download it separately before running the project.

---

## Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/uk-sme-import-margin-simulator.git
cd uk-sme-import-margin-simulator
```

---

### 2. Create and Activate Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate      # Mac/Linux
# or
venv\Scripts\activate         # Windows
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Download Required Data Files

1. Go to the Google Drive folder:
   https://drive.google.com/drive/folders/1LN8HQftFSFMf2rlNm3GLs-iwxXlZpp2J?usp=sharing

2. Download the dataset ZIP file

3. Extract (unzip) the files

4. Move the extracted `data/` folder into the project root:

```
uk-sme-import-margin-simulator/
├── app.py
├── scripts/
├── tests/
├── data/   ← PLACE HERE
```

 Do NOT rename or modify the folder structure.

---

### 5. Run the Application

```bash
streamlit run app.py
```

The dashboard will open in your browser (usually at http://localhost:8501).

---

## Testing

Run the full test suite:

```bash
pytest --cov
```

*  330+ tests
*  ~81% coverage
* Includes unit, feature, UI, and integration tests

---

## Data Sources

* HMRC – UK Trade Data
* ONS – Office for National Statistics
* Bank of England – Exchange Rates

---

## Documentation

* `QA_Report.html` → Testing & QA report
* `WORKFLOW.md` → Architecture and system workflow

---

## Key Learning Outcomes

* Financial modelling and simulation
* Time series forecasting (ARIMA)
* Risk analytics