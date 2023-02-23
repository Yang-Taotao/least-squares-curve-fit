"""
This Python script performs a linear least squares fit with SciPy
It fit some x,y dataset generated with NumPy.
It extracts the error, covariance, and correlaation coeff of the fit.
It then performs a chi-square test with SciPy to find the p-value.
The script then print out the results in text form.
"""

# General library import
# NumPy import for array manipulation, aliased as np
import numpy as np

# SciPy import of curve_fit module for curve fitting
from scipy.optimize import curve_fit

# SciPy import of chi2 module for chi-square calculation
from scipy.stats import chi2

# Generate x,y data
# Generate an array with np.array() called data_x with data entry of x
data_x = np.array([2.71, 2.05, 2.67, 2.23, 2.36, 2.52, 2.91, 2.43])
# Generate an array with np.array() called data_y with data entry of y
data_y = np.array([-21.1, -19.2, -20.6, -19.4, -20.0, -20.2, -21.5, -19.8])
# Input known standard deviation of the residual term as data_sigma
data_sigma = 0.17

# Define a linear fit function with the style of y = a+b*x


def fit_func(fit_x, fit_a: float, fit_b: float):
    """
    Defines a linear function in the style of y = a+b*x

    Parameters:
        fit_x: x value array
        fit_a: Intercept term
        fit_b: Slope term

    Returns:
        fit_func: A customized linear fit function, or y value array
    """
    return fit_a + fit_b * fit_x


# Curve fitting
# Perform a linear least squared fit with curve_fit() method
# Generate the fit parameters array as fit_param
# Generate the covariance matrix of the fit as fit_covar
fit_param, fit_covar = curve_fit(fit_func, data_x, data_y)
# Unpack fit parameters into intercept as fit_inter and slope as fit_slope
fit_inter, fit_slope = fit_param
# Pass the fitted result to construct the fitted function as fit_model
fit_model = fit_inter + fit_slope * data_x
# Extract the variance from the diagnal terms of covariance matrix
# The diagnal term is extracted with np.diag() method
# The variance value array is stored as fit_varia
fit_varia = np.diag(fit_covar)
# Compute the standard deviation from the variance
# The np.sqrt() method compute the square root of the variance
# The standard deviation value array is stored as fit_sigma
fit_sigma = np.sqrt(fit_varia)
# Compute the correlation coeff matrix from data set entry
# Index the correlation coefficient from the correlation coeff matrix
# The correlation coefficient is stored as cor_coeff
cor_coeff = np.corrcoef(data_x, data_y)[0, 1]

# Chi-square test
# Compute the dof from the number of data entries and fit parameters
# The number of data entries is calculated from its length with len()
# The number of fit parameters is calculated from the length with len()
# The degrees of freedom is stored as dof_val
dof_val = len(data_x) - len(fit_param)
# Compute chi-square result from the residual term and the known sigma
# The chi-square value is the sum of terms with np.sum()
# The chi-square value is stored as chi_sqr
chi_sqr = np.sum(((data_y - fit_model) / data_sigma) ** 2)
# Compute p-value based on chi-square result and degrees of freedom
# Using chi2.cdf() method to construct a cumulative distribution function
# The cdf is based on the chi-square value and degrees of freedom
# The p-value is stored as p-val
p_val = 1 - chi2.cdf(chi_sqr, dof_val)

# Result printout
# Prints are formatted with f-strings and rounded to three decimal places
# Headers are left justified and values are right justified
# An empty line is printed with print() to add space between the outputs
# Print the fitted parameters
print(f"{'Fitted parameters':<20}")
print(f"{'Slope:':<20}{fit_slope:>10.3f}")
print(f"{'Intercept:':<20}{fit_inter:>10.3f}")
print()
# Print the standard deviation of the fit parameters
print(f"{'Fitted standard deviation':<20}")
print(f"{'Slope:':<20}{fit_sigma[1]:>10.3f}")
print(f"{'Intercept:':<20}{fit_sigma[0]:>10.3f}")
print()
# Print the covariance of the fit parameters
print(f"{'Fitted covariance':<20}")
print(f"{'Slope:':<20}{fit_covar[1,1]:>10.3f}")
print(f"{'Intercept:':<20}{fit_covar[0,0]:>10.3f}")
print()
# Print the overall covariance and correlation coefficient of the fit
print(f"{'Overall covariance; Overall correlation coefficient':<20}")
print(f"{'Covariance:':<20}{fit_covar[0,1]:>10.3f}")
print(f"{'Coefficient:':<20}{cor_coeff:>10.3f}")
print()
# Print the chi-square test result and p-value
print(f"{'Chi-square test result':<20}")
print(f"{'Chi-square:':<20}{chi_sqr:>10.3f}")
print(f"{'dof:':<20}{dof_val:>10.3f}")
print(f"{'p-val:':<20}{p_val:>10.3f}")
