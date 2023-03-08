"""
This Python script performs a linear least squares fit.
It generates some x,y dataset with NumPy.
It extracts the variance, covariance, and correlaation coefficient.
It then performs a chi-square test with SciPy to find the p-value.
The script then print out the results in text form.

Created on Thu Mar  2 2023

@author: Yang-Taotao
"""

# General library import
# Import numpy with alias np for array manipulation
import numpy as np

# Import chi2 module from scipy.stats for chi-square calculation
from scipy.stats import chi2

# Generate x,y data arrays with np.array() method
# Generate x data as data_x
data_x = np.array([2.71, 2.05, 2.67, 2.23, 2.36, 2.52, 2.91, 2.43])
# Generate y data as data_y
data_y = np.array([-21.1, -19.2, -20.6, -19.4, -20.0, -20.2, -21.5, -19.8])
# Define the known standard deviation of the residual term as data_sigma
data_sigma = 0.17
# Calculate sample size from the number of x entries with len() method
# Assign the sample size as data_n
data_n = len(data_x)

# Intermidiate calculations
# Calculate the sum of terms using np.sum() method
# Calculate the square root of terms using np.sqrt() method
# Calculate the sum of all x data as sum_x
sum_x = np.sum(data_x)
# Calculate the sum of all y data as sum_y
sum_y = np.sum(data_y)
# Calculate the sum of the squared value of all x data as sum_x_sqr
sum_x_sqr = np.sum(data_x**2)
# Calculate the sum of the squared value of all y data as sum_y_sqr
sum_y_sqr = np.sum(data_y**2)
# Calculate the sum of the product of all x and y data as sum_xy
sum_xy = np.sum(data_x * data_y)
# Generate the denominator for least square fitting calculation as denom_ls
denom_ls = data_n * sum_x_sqr - (sum_x) ** 2
# Generate the denominator for the correlation coefficient calculations
# Generate the x related denominator as denom_corr_x
denom_corr_x = np.sqrt(data_n * sum_x_sqr - (sum_x) ** 2)
# Generate the y related denominator as denom_corr_y
denom_corr_y = np.sqrt(data_n * sum_y_sqr - (sum_y) ** 2)

# Linear least squares fitting with form y=als+bls*x
# Calculate the intercept of the least square fit from its definition
# Assign the fitted intercept value as fit_als
fit_als = (sum_y * sum_x_sqr - sum_xy * sum_x) / denom_ls
# Calculate the slope of the least square fit from its definition
# Assign the fitted slope value as fit_bls
fit_bls = (data_n * sum_xy - sum_y * sum_x) / denom_ls
# Collect the fitted intercept and slope values into a tuple as fit_param
fit_param = fit_als, fit_bls
# Calculate the variance of the fitted terms from their definition
# Calculate the variance of intercept term as fit_var_als
fit_var_als = ((data_sigma**2) * sum_x_sqr) / denom_ls
# Calculate the variance of slope term as fit_var_bls
fit_var_bls = ((data_sigma**2) * data_n) / denom_ls
# Calculate the standard deviation from the variance results
# Calculate the standard deviation of the intercept as fit_sigma_als
fit_sigma_als = np.sqrt(fit_var_als)
# Calculate the standard deviation of the slope as fit_sigma_bls
fit_sigma_bls = np.sqrt(fit_var_bls)
# Calculate the covariance of the intercept and slope from definition
# Assign the covariance result as fit_cov_alsbls
fit_cov_alsbls = (-(data_sigma**2) * sum_x) / denom_ls
# Calculate the correlation coefficient from definition as fit_corr_xy
fit_corr_xy = (data_n * sum_xy - sum_x * sum_y) / (denom_corr_x * denom_corr_y)

# Construct linear model from fitted slope and intercept
# Assign x entries of the model from original data as model_x
model_x = data_x
# Generate y entries of the model from the x data of the model as model_y
model_y = fit_als + fit_bls * model_x

# Chi-square, degree of freedom (dof), and p-value calculations
# Calcualte the chi-square value from definition as chi_sqr
chi_sqr = np.sum(((data_y - model_y) / (data_sigma)) ** 2)
# Calculate the dof from sample size and number of fit parameters
# Assign degree of freedom as chi_dof
chi_dof = data_n - len(fit_param)
# Calculate p-value with chi2.cdf() method as chi_p_val
# Construct chi-square cdf with chi2.cdf() method
# Use calculated chi-square value and dof to generate p-value
chi_p_val = 1 - chi2.cdf(chi_sqr, chi_dof)

# Results print out
# Format prints with f-string methods
# The labels are left justified with cell size of 20 print(f"{}:>20")
# The numbers are right justified with cell size of 10 with print(f"{}:>10")
# The numbers are left with three decimal places with print(f"{}:.3f")
# Print an empty line at the end of each print out subsection with print()
# Print the fitted intercept and slope
print(f"{'Fitted intercept and slope':<20}")
print(f"{'Intercept:':<20}{fit_als:>10.3f}")
print(f"{'Slope:':<20}{fit_bls:>10.3f}")
print()
# Print the variance of the fit parameters
print(f"{'Variance of the fitted intercept and slope':<20}")
print(f"{'Intercept:':<20}{fit_var_als:>10.3f}")
print(f"{'Slope:':<20}{fit_var_bls:>10.3f}")
print()
# Print the standard deviation of the fit parameters
print(f"{'Standard deviation of the fitted intercept and slope':<20}")
print(f"{'Intercept:':<20}{fit_sigma_als:>10.3f}")
print(f"{'Slope:':<20}{fit_sigma_bls:>10.3f}")
print()
# Print the covariance of the fit parameters
print(f"{'Covariance and correlation coefficient of the fit':<20}")
print(f"{'Covariance:':<20}{fit_cov_alsbls:>10.3f}")
print(f"{'Coefficient:':<20}{fit_corr_xy:>10.3f}")
print()
# Print the chi-square test result and p-value
print(f"{'Chi-square test result':<20}")
print(f"{'Chi-square:':<20}{chi_sqr:>10.3f}")
print(f"{'Degree of freedom:':<20}{chi_dof:>10.3f}")
print(f"{'p-value:':<20}{chi_p_val:>10.3f}")
