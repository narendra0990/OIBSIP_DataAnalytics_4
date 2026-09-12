# 🏡 Predicting House Prices with Linear Regression & Regularization

**Organization:** Oasis Infobyte (OIBSIP)  
**Track:** Data Analytics (Level 2 • Task 1)  
**Author:** Narendra

---

## 📌 Project Overview
This project formulates, trains, and evaluates econometric regression models predicting residential property transaction values based on physical and geographic attributes. The pipeline benchmarks Ordinary Least Squares (OLS) against Ridge ($L_2$) and Lasso ($L_1$) regularized estimators.

---

## 📁 Repository Structure
```text
├── data/
│   └── housing_data.csv
├── notebooks/
│   └── house_price_prediction.ipynb
├── charts/
│   ├── actual_vs_predicted_prices.png
│   ├── feature_coefficients_impact.png
│   ├── housing_correlation_heatmap.png
│   └── residual_analysis_diagnostics.png
├── ml_core.py
├── run_regression.py
├── requirements.txt
└── README.md
```

---

## 📈 Model Performance & Econometric Coefficients
- **Accuracy**: Achieved **$R^2 = 0.9706$** ($MAE = \$13,750$).
- **Living Area**: Adds **+\$145.00** per additional square foot.
- **Location Premium**: `Lakeside` (+**\$85k**) and `Downtown` (+**\$65k**) command highest valuation premiums.
- **Depreciation**: Housing depreciates at **-\$650/year** of age and **-\$1,800/km** from city center.

---

## 🚀 How to Run Locally
```bash
pip install -r requirements.txt
python run_regression.py
jupyter notebook notebooks/house_price_prediction.ipynb
```
