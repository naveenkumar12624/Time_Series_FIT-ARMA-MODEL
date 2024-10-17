# Ex.No:04 FIT ARMA MODEL FOR TIME SERIES
### Date: 
### Name: Naveen Kumar S
### Reg No: 212221240033

### AIM:
To implement ARMA model in python.

### ALGORITHM:
Step 1: Load and prepare the NVIDIA stock prices time series data into a variable.

Step 2: Plot the time series data using Matplotlib with appropriate titles and labels.

Step 3: Calculate and plot the Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) using statsmodels.

Step 4: Fit an ARMA model to the time series data using the specified order (p, d, q) and store the fitted model.

Step 5: Visualize the original time series data alongside the fitted values from the ARMA model in a new plot.

### PROGRAM:
```python
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# Load dataset
file_path = 'C:/Users/lenovo/Downloads/archive (2)/NVIDIA/NvidiaStockPrice.csv'  # Update with your actual file path
data = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')

# Extract 'Close' prices
series = data['Close']
```
```py
# Plot the time series data
plt.figure(figsize=(10, 6))
plt.plot(series)
plt.title('NVIDIA Stock Prices')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.show()
```
```py
# Plot ACF and PACF
plt.figure(figsize=(12, 6))
plt.subplot(121)
plot_acf(series, ax=plt.gca())
plt.title('Autocorrelation Function (ACF)')

plt.subplot(122)
plot_pacf(series, ax=plt.gca())
plt.title('Partial Autocorrelation Function (PACF)')
plt.show()
```
```py
# Fit ARMA model
# Note: For ARMA, d=0 (difference order should be 0 for ARMA)
model = ARIMA(series, order=(2, 0, 2))  # Adjust p, d, q values based on ACF and PACF plots
model_fit = model.fit()

# Print model summary
print(model_fit.summary())

# Plot the fitted values
plt.figure(figsize=(10, 6))
plt.plot(series, label='Original Series')
plt.plot(model_fit.fittedvalues, color='red', label='Fitted Values')
plt.title('Original vs Fitted Values')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()
```


### OUTPUT:

# Partial Autocorrelation
![download](https://github.com/user-attachments/assets/9801a160-399d-4209-96a7-a739d2625fcd)

# Autocorrelation
![download](https://github.com/user-attachments/assets/399c5815-335a-4fed-addb-2bd7c9697833)

# SARIMAX Results:
![image](https://github.com/user-attachments/assets/ba86c69c-9e5b-4014-aee9-62f1107a594a)

# Original vs Fitted Value:
![download](https://github.com/user-attachments/assets/8589b6c2-0eee-4c8b-a232-a3bedf9d4526)

### RESULT:
Thus, the python program is created to fit ARMA Model successfully.
