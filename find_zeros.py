#!/usr/bin/env python3
"""
Find zeros of the derivative f'(x) = 6x - 4 over the range [-4, 4]
Visualizes the numerical derivative results from Lean
"""

import numpy as np
import matplotlib.pyplot as plt

# Define the function and its derivative
def f(x):
    """f(x) = 3x² - 4x + 5"""
    return 3*x**2 - 4*x + 5

def df_dx(x):
    """Analytical derivative: f'(x) = 6x - 4"""
    return 6*x - 4

def numerical_derivative(x, h=1e-8):
    """Numerical derivative using finite differences"""
    return (f(x + h) - f(x)) / h

# Create range from -4 to 4
x_range = np.linspace(-4, 4, 100)
x_test_points = [-4, -3, -2, -1, 0, 0.5, 0.6, 0.66, 0.667, 0.67, 0.7, 1, 2, 3, 4]

print("🔍 FINDING ZEROS OF THE DERIVATIVE f'(x) = 6x - 4")
print("=" * 60)
print("Function: f(x) = 3x² - 4x + 5")
print("Derivative: f'(x) = 6x - 4")
print("\nEvaluating numerical derivative (delx) at key points:")
print("-" * 60)

# Evaluate at key points to find zero
for x in x_test_points:
    numerical = numerical_derivative(x)
    analytical = df_dx(x)
    print(f"x = {x:6.3f} | delx = {numerical:8.5f} | f'(x) = {analytical:8.5f}")

# Find exact zero
zero_point = 4/6  # Exact solution to 6x - 4 = 0
print(f"\n🎯 EXACT ZERO FOUND:")
print(f"x = 2/3 = {zero_point:.6f}")
print(f"delx({zero_point:.6f}) = {numerical_derivative(zero_point):.8f}")
print(f"f'({zero_point:.6f}) = {df_dx(zero_point):.8f}")
print(f"f({zero_point:.6f}) = {f(zero_point):.6f} (minimum value)")

# Create visualization
plt.figure(figsize=(12, 8))

# Plot the function
plt.subplot(2, 1, 1)
x_smooth = np.linspace(-4, 4, 400)
plt.plot(x_smooth, f(x_smooth), 'b-', linewidth=2, label='f(x) = 3x² - 4x + 5')
plt.plot(zero_point, f(zero_point), 'ro', markersize=8, label=f'Minimum at x = {zero_point:.3f}')
plt.grid(True, alpha=0.3)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Function f(x) = 3x² - 4x + 5')
plt.legend()

# Plot the derivative
plt.subplot(2, 1, 2)
plt.plot(x_smooth, df_dx(x_smooth), 'g-', linewidth=2, label="f'(x) = 6x - 4 (analytical)")
plt.plot(x_test_points, [numerical_derivative(x) for x in x_test_points], 'ro', 
         markersize=6, label='delx (numerical)')
plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
plt.axvline(x=zero_point, color='r', linestyle='--', alpha=0.5, 
            label=f'Zero at x = {zero_point:.3f}')
plt.grid(True, alpha=0.3)
plt.xlabel('x')
plt.ylabel("f'(x)")
plt.title("Derivative f'(x) = 6x - 4 (Finding where derivative = 0)")
plt.legend()

plt.tight_layout()
plt.savefig('derivative_zeros.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\n📊 Visualization saved as 'derivative_zeros.png'")
print(f"\n✅ CONCLUSION:")
print(f"   The derivative f'(x) = 6x - 4 has exactly ONE zero at x = 2/3 ≈ 0.6667")
print(f"   This corresponds to the minimum of f(x) at the point ({zero_point:.3f}, {f(zero_point):.3f})")