# **Risk Management System**

 A framework for analyzing portfolio risk with metrics like VaR, CVaR, and Maximum Drawdown. Includes a stress test (COVID-19 crash, and Monte Carlo simulations to assess diversification and portfolio resilience. 

---

## **Features**
- **Risk Metrics**:
  - **Value at Risk (VaR)**: Historical, Gaussian, and Modified Gaussian methods.
  - **Conditional VaR (CVaR)**: Historical and Gaussian methods to evaluate tail risks.
  - **Maximum Drawdown (MDD)**: Identifies the worst portfolio declines and recovery times.

- **Stress Testing**:
  - Analyze portfolio performance during the **COVID-19 crash** (Feb–Mar 2020).
  - Simulate **hypothetical market shocks**, such as a 10% equity crash.

- **Monte Carlo Simulations**:
  - Generate thousands of random return paths to assess portfolio risk under various scenarios.

- **Diversification Analysis**:
  - Evaluate the impact of diversification on reducing portfolio risk using covariance and correlation matrices.

---

## **Data**
- **Portfolio Composition**:
  - AAPL (Tech)
  - JNJ (Healthcare)
  - XOM (Energy)
  - TLT (Bonds ETF)
  - GLD (Gold ETF)
  
- **Portfolio Weights**:
  - AAPL: 25%, JNJ: 20%, XOM: 25%, TLT: 15%, GLD: 15%.
  
- **Historical Data**:
  - Daily adjusted closing prices over 5 years (e.g., 2018–2023).

---

## **Technologies**
- **Python Libraries**:
  - `pandas` for data manipulation.
  - `numpy` for numerical operations.
  - `matplotlib` for visualizations.
  - `scipy.stats` for statistical analysis.
  
---

## **Usage**
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/risk-management-system.git

2. Install required dependencies::
   ```bash
   pip install -r requirements.txt
   
3. Run the Jupyter Notebook or script:
   ```bash
   jupyter main.ipynb


## **Future Enhancements**

 * Include machine learning models for risk prediction.
 * Add support for multi-asset class portfolios (e.g., crypto, real estate).
 * Automate data fetching using APIs like Yahoo Finance.t
