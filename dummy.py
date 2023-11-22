import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Function to simulate data
def generate_data(n=100):
    np.random.seed(42)  # For reproducibility
    x1 = np.random.rand(n) * 10
    x2 = np.random.rand(n) * 10
    x3 = np.random.rand(n) * 10
    # Simulating a nonlinear relationship
    median = 1000 + (500 * np.exp(0.05 * x1 - 0.02 * x2 + 0.03 * x3)) + np.random.normal(0, 100, n)
    return pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3, 'median': median})

# Generate sample data
df = generate_data()

# Example nonlinear model
def model(params, x1, x2, x3):
    a, b, c, d, e = params
    return a * np.exp(b * x1 + c * x2 + d * x3) + e

# Objective function to minimize (sum of squared errors)
def objective_function(params, x1, x2, x3, y):
    return np.sum((model(params, x1, x2, x3) - y) ** 2)

# Initial guess for parameters
initial_guess = [1000, 0.05, -0.02, 0.03, 100]

# Perform minimization
result = minimize(objective_function, initial_guess, args=(df['x1'], df['x2'], df['x3'], df['median']))

# Resulting parameters
fitted_params = result.x
print("Fitted Parameters:", fitted_params)

# Adding predictions to the DataFrame
df['predicted_median'] = model(fitted_params, df['x1'], df['x2'], df['x3'])

# Plotting the results
plt.figure(figsize=(10, 6))
plt.scatter(df['median'], df['predicted_median'], alpha=0.7)
plt.plot([df['median'].min(), df['median'].max()], [df['median'].min(), df['median'].max()], 'k--')  # Diagonal line
plt.xlabel('Actual Median')
plt.ylabel('Predicted Median')
plt.title('Actual vs. Predicted Median Values')
plt.show()
