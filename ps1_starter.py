import numpy as np
from numpy.random import randn
import matplotlib.pyplot as plt

#Problem 1
def compute_slope_estimator(x_vals,y_vals):
    tot = 0
    div = 0
    n = len(x_vals)
    mean_x = sum(x_vals)/n
    mean_y = sum(y_vals)/n
    for i in range(n):
        tot += x_vals[i]*y_vals[i]
        div += x_vals[i]**2
    return (tot - n*mean_x*mean_y) / (div - n*mean_x**2)

#Problem 2
def compute_intercept_estimator(x_vals,y_vals):
    n = len(x_vals)
    mean_x = sum(x_vals)/n
    mean_y = sum(y_vals)/n
    return mean_y - compute_slope_estimator(x_vals,y_vals)*mean_x

#Problem 3
def train_model(x_vals,y_vals):
    #your code here
    return (a,b)

#Problem 4
def dL_da(x_vals,y_vals,a,b):
    pass

#Problem 5
def dL_db(x_vals,y_vals,a,b):
    pass

#Problem 6
def gradient_descent_step(x_vals,y_vals,a,b,k=0.01):
    pass

#Problem 7
def gradient_descent(x_vals,y_vals,a_0=0,b_0=0,k=1000):
    pass

# Problem 8
def fit_quadratic(x_vals, y_vals):
    return (a, b)

# Problem 9
def calculate_scaling_parameters(d_vals, l_vals):
    return (a, b)

## Example values for Problem 9

# number of training tokens
d_vals = [2 ** i for i in range(24,34)]
# cross-entropy loss for each model in the example
l_vals = [4.00, 3.95, 3.55, 3.43, 3.12, 3.00, 2.79, 2.50, 2.35, 2.22]

## Additional functions

def generate_y_vals(x_vals, a=1, b=0, std_dev=0, f=lambda x: x, g_inverse=lambda y: y):
    """
    Generates noisy output data where g(y) has a linear relationship with f(x).

    Parameters:
    x_vals (numpy array): The observed values of the independent variable x.
    a (float): Scaling factor applied to f(x). Default is 1.
    b (float): Bias or intercept term added to a*f(x). Default is 0.
    std_dev (float): Standard deviation of the normally distributed noise added to a*f(x)+b. Default is 0 (no noise).
    f (function): Transformation applied to x. Default is the identity function (no transformation).
    g_inverse (function): The inverse of the transformation applied to y. Default is the identity function (no transformation).

    Returns:
    y_vals (numpy array): The noisy observed output values.
    """
    # Number of observations (length of the input array x_obs)
    n = len(x_vals)

    # Generate normally distributed noise with mean 0 and standard deviation `std_dev`
    errors = np.random.normal(0, std_dev, size=n)

    # Compute the output y using the formula: y = g_inverse(a * f(x) + b + errors)
    y_vals = g_inverse(a * f(x_vals) + b + errors)

    return y_vals

def plot_generated_data(x_vals, y_vals, scaled=False, f=lambda x: x, g=lambda y: y):
    """
    Plots the data, either transformed (if scaled=True) or untransformed (if scaled=False),
    allowing a linear relationship to be visualized when transformations are applied.

    Parameters:
    x_vals (numpy array): The observed values of the independent variable x.
    y_vals (numpy array): The observed values of the dependent variable y.
    scaled (bool): If True, plot the transformed data (f(x) vs g(y)), otherwise plot untransformed data (x vs y).
    f (function): Transformation applied to x. Default is the identity function (no transformation).
    g (function): Transformation applied to y. Default is the identity function (no transformation).
    """
    if scaled:
        # Apply the transformations f(x) and g(y)
        transformed_x = f(x_vals)
        transformed_y = g(y_vals)
        xlabel = 'f(x)'
        ylabel = 'g(y)'
        title = 'Transformed data: g(y) vs f(x)'
    else:
        # Use untransformed data
        transformed_x = x_vals
        transformed_y = y_vals
        xlabel = 'x'
        ylabel = 'y'
        title = 'Original data: y vs x'

    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.scatter(transformed_x, transformed_y, label='Data', color='b', s=10)

    # Add labels and title
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    # Display the plot
    plt.show()
