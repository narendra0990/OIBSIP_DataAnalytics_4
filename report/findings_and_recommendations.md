# 🏡 House Price Prediction — Econometric Findings & Valuation Report

**Domain:** Data Analytics  
**Organization:** Oasis Infobyte (OIBSIP)  
**Project:** Level 2 • Task 1  
**Author:** Narendra  

---

## 🔍 Model Performance Benchmark

| Model Architecture | MAE ($) | RMSE ($) | $R^2$ Test Score |
| :--- | :--- | :--- | :--- |
| **Ordinary Least Squares (OLS)** | \$13,750.40 | \$17,025.09 | **0.9706** |
| **Ridge Regression ($L_2$, $\alpha=1.0$)** | \$13,715.62 | \$16,961.49 | **0.9708** |
| **Lasso Regression ($L_1$, $\alpha=0.1$)** | \$13,750.32 | \$17,024.94 | **0.9706** |

---

## 🔑 Econometric Insights & Marginal Coefficients

1. **Living Area (Square Footage)**:
   - Contributes **+\$145.00** in property valuation per additional square foot (strongest structural driver).
2. **Neighborhood Premiums**:
   - `Lakeside` (+**\$85,000**) and `Downtown` (+**\$65,000**) command massive valuation premiums relative to baseline suburbs.
3. **Depreciation & Distance Penalty**:
   - Each year of house age decreases valuation by **-\$650/year**.
   - Each kilometer farther from the city center decreases valuation by **-\$1,800/km**.
4. **Diagnostic Integrity**:
   - Parity plot shows near-perfect diagonal fit.
   - Residual errors are homoscedastic and normally distributed with zero mean.
