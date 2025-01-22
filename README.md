# Investment Portfolio Optimization

## Overview
This project demonstrates **Investment Portfolio Optimization** using **Modern Portfolio Theory (MPT)** and **Monte Carlo Simulations**. It helps investors construct a stock portfolio that maximizes returns while minimizing risk.

The project includes:
- Data collection using Yahoo Finance.
- Risk-return analysis.
- Portfolio optimization using Monte Carlo simulations.
- Efficient Frontier visualization.
- Optimal portfolio selection (Maximum Sharpe Ratio & Minimum Volatility Portfolio).

## Features
- **Stock Data Retrieval**: Fetches historical stock prices for selected Australian stocks (ASX-listed).
- **Risk & Return Calculation**: Computes annualized returns and the covariance matrix.
- **Monte Carlo Simulations**: Generates 10,000 random portfolios and evaluates their risk-return characteristics.
- **Optimal Portfolio Selection**:
  - Maximum Sharpe Ratio Portfolio
  - Minimum Volatility Portfolio
- **Data Export**:
  - Results saved as an **Excel file** with separate sheets for all portfolios, optimal portfolios, and risk metrics.
  - **Efficient Frontier plot** showing optimal portfolio points.

## Dependencies
Ensure you have the following Python libraries installed:
```sh
pip install numpy pandas matplotlib yfinance
```

## Usage
Run the script:
```sh
python portfolio_optimisation.py
```

The script will:
1. Download historical stock prices.
2. Calculate key portfolio metrics.
3. Perform Monte Carlo simulations.
4. Generate an **Excel file** with the results.
5. Save the **Efficient Frontier plot**.

## Output Files
- **Portfolio_Optimization_Results.xlsx**: Contains portfolio risk-return metrics and optimal weights in percentages.
- **Portfolio_Optimization_Plot.png**: Efficient Frontier visualization with clearly marked optimal portfolios.

## Conceptual Background
### 1. **Modern Portfolio Theory (MPT)**
MPT, developed by **Harry Markowitz**, focuses on risk-optimized portfolio allocation. It suggests that portfolios should be designed based on:
- **Expected Return**: Weighted sum of individual asset returns.
- **Risk (Volatility)**: Portfolio standard deviation.
- **Sharpe Ratio**: Measures return per unit of risk.

### 2. **Monte Carlo Simulations**
Monte Carlo simulations generate thousands of portfolio combinations by randomly assigning weights to assets. This helps in identifying the:
- **Efficient Frontier**: The set of optimal portfolios that offer the highest return for a given risk.
- **Optimal Portfolio**: The portfolio with the **highest Sharpe Ratio**.
- **Minimum Volatility Portfolio**: The portfolio with the lowest risk.

## Project Structure
```
📂 Investment-Portfolio-Optimization
│── portfolio_optimisation.py  # Main script
│── Portfolio_Optimization_Results.xlsx  # Output Excel file
│── Portfolio_Optimization_Plot.png  # Efficient Frontier plot
│── README.md  # Project documentation
```

## Potential Enhancements
- **Additional Asset Classes**: Include bonds, ETFs, and cryptocurrencies.
- **Factor-Based Optimization**: Implement CAPM-based portfolio selection.
- **Live Portfolio Updates**: Automate portfolio rebalancing with real-time data.

## Author
**Prateek Bose**  
Master of Management (Finance)  
University of Melbourne  

---
Feel free to contribute or raise issues in this repository!
