# Ex.No:04 FIT ARMA MODEL FOR TIME SERIES
### Date: 
### Name: Syed Abdul Wasih H
### Reg No: 212221240057

### AIM:
To implement ARMA model in python.

### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000 data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.
4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000 data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.
6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using plot_acf and plot_pacf.

### PROGRAM:
```python
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# 1. Import necessary libraries (already done above)

# 2. Set up matplotlib settings for figure size
plt.rcParams['figure.figsize'] = (12, 6)

# 3. Define an ARMA(1,1) process and generate 1000 data points
# ARMA(1,1) with coefficients ar1 = 0.5 and ma1 = 0.3
ar1 = np.array([1, -0.5])  # Note: AR coefficients are defined with a leading 1 for the model equation
ma1 = np.array([1, 0.3])   # MA coefficients similarly start with 1

arma11_process = ArmaProcess(ar1, ma1)
sample_arma11 = arma11_process.generate_sample(nsample=1000)

# Plot the generated time series for ARMA(1,1)
plt.plot(sample_arma11)
plt.title('Simulated ARMA(1,1) Process')
plt.xlim(0, 1000)
plt.show()

# 4. Display the autocorrelation and partial autocorrelation plots for ARMA(1,1)
plot_acf(sample_arma11, lags=30)
plt.title('Autocorrelation of ARMA(1,1) Process')
plt.show()

plot_pacf(sample_arma11, lags=30)
plt.title('Partial Autocorrelation of ARMA(1,1) Process')
plt.show()

# 5. Define an ARMA(2,2) process and generate 10000 data points
# ARMA(2,2) with coefficients ar2 = [0.75, -0.25] and ma2 = [0.65, 0.35]
ar2 = np.array([1, -0.75, 0.25])
ma2 = np.array([1, 0.65, 0.35])

arma22_process = ArmaProcess(ar2, ma2)
sample_arma22 = arma22_process.generate_sample(nsample=10000)

# Plot the generated time series for ARMA(2,2)
plt.plot(sample_arma22)
plt.title('Simulated ARMA(2,2) Process')
plt.xlim(0, 10000)
plt.show()

# 6. Display the autocorrelation and partial autocorrelation plots for ARMA(2,2)
plot_acf(sample_arma22, lags=30)
plt.title('Autocorrelation of ARMA(2,2) Process')
plt.show()

plot_pacf(sample_arma22, lags=30)
plt.title('Partial Autocorrelation of ARMA(2,2) Process')
plt.show()
```

### OUTPUT:
# SIMULATED ARMA(1,1) PROCESS:
![image](https://github.com/user-attachments/assets/c0b78dac-2513-4dde-8806-662ff72dc322)

# Partial Autocorrelation
![image](https://github.com/user-attachments/assets/4b865320-1db6-4bb5-bdc2-aee06044c09b)

# Autocorrelation
![image](https://github.com/user-attachments/assets/8fc25cb3-68a5-469f-8da1-e5860df861f2)

# SIMULATED ARMA(2,2) PROCESS:
![image](https://github.com/user-attachments/assets/ac7dfb18-7fa2-4a45-b031-091867f6af38)

# Partial Autocorrelation
![image](https://github.com/user-attachments/assets/0cfef4cf-cfa0-438b-ba56-4f6b564e4b68)

# Autocorrelation
![image](https://github.com/user-attachments/assets/1c740a47-cec5-409d-ad25-ac728ec78d3f)


### RESULT:
Thus, the python program is created to fit ARMA Model successfully.
