import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Define ASX stock symbols (You can change these)
stocks = ['CBA.AX', 'MQG.AX', 'NAB.AX', 'XRO.AX', 'BHP.AX']  

# Fetch stock data (only 'Close' prices)
data = yf.download(stocks, start="2020-01-01", end="2024-01-01")['Close']

# Display first 5 rows
print(data.head())

# Compute log returns
returns = np.log(data / data.shift(1))

# Compute annualized returns (252 trading days in a year)
mean_returns = returns.mean() * 252

# Compute annualized covariance matrix (Risk)
cov_matrix = returns.cov() * 252

# Display calculated metrics
print("\nAnnualized Returns:")
print(mean_returns)

print("\nCovariance Matrix:")
print(cov_matrix)

#_____________________________________________________________________________________

num_portfolios = 10000  # Generate 10,000 portfolios
num_stocks = len(stocks)

# Generate random weights for each portfolio
all_weights = np.random.dirichlet(np.ones(num_stocks), num_portfolios)

# Compute expected returns & risk (volatility)
returns_arr = np.sum(mean_returns.values * all_weights, axis=1)
volatility_arr = np.sqrt(np.einsum('ij,jk,ik->i', all_weights, cov_matrix.values, all_weights))

# Compute Sharpe Ratio (Higher is better)
sharpe_arr = returns_arr / volatility_arr

# Find Portfolio with Maximum Sharpe Ratio
max_sharpe_idx = sharpe_arr.argmax()
optimal_weights = all_weights[max_sharpe_idx]

# Find Portfolio with Minimum Volatility
min_vol_idx = volatility_arr.argmin()
min_vol_weights = all_weights[min_vol_idx]

#_____________________________________________________________________________________

# Create a DataFrame for visualization
portfolio_results = pd.DataFrame({
    "Expected Return": returns_arr,
    "Volatility (Risk)": volatility_arr,
    "Sharpe Ratio": sharpe_arr
})

# Format and save to Excel
output_file = "Portfolio_Optimization_Results.xlsx"
with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
    workbook = writer.book
    
    # Save portfolio results
    portfolio_results.to_excel(writer, sheet_name='Portfolios', index=False)
    worksheet = writer.sheets['Portfolios']
    format1 = workbook.add_format({'num_format': '0.0000', 'align': 'center'})
    worksheet.set_column('A:C', 18, format1)
    
    # Save optimal portfolios
    optimal_df = pd.DataFrame(optimal_weights * 100, index=stocks, columns=['Optimal Weights (%)'])
    min_vol_df = pd.DataFrame(min_vol_weights * 100, index=stocks, columns=['Min Volatility Weights (%)'])

    optimal_df.to_excel(writer, sheet_name='Optimal Portfolio', index=True)
    worksheet = writer.sheets['Optimal Portfolio']
    worksheet.set_column('A:B', 25, format1)
    
    min_vol_df.to_excel(writer, sheet_name='Min Volatility Portfolio', index=True)
    worksheet = writer.sheets['Min Volatility Portfolio']
    worksheet.set_column('A:B', 25, format1)

#_____________________________________________________________________________________    

# Generate a detailed and professional plot
plt.figure(figsize=(10, 6))
plt.scatter(volatility_arr, returns_arr, c=sharpe_arr, cmap='viridis', alpha=0.5)
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatility (Risk)')
plt.ylabel('Expected Return')
plt.title('Efficient Frontier - Portfolio Optimization')

# Highlight optimal portfolios
plt.scatter(volatility_arr[max_sharpe_idx], returns_arr[max_sharpe_idx], c='red', marker='*', s=200, label='Max Sharpe Ratio')
plt.scatter(volatility_arr[min_vol_idx], returns_arr[min_vol_idx], c='blue', marker='X', s=200, label='Min Volatility')
plt.legend()

# Save the figure
plot_file = "Portfolio_Optimization_Plot.png"
plt.savefig(plot_file, dpi=300, bbox_inches='tight')

# Display a confirmation message
print(f"Portfolio optimization complete. Results saved to {output_file} and figure saved to {plot_file}.")