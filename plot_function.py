import numpy as np
import matplotlib.pyplot as plt

def f(x):
    """The function f(x) = 3x² - 4x + 5 from the Lean file"""
    return 3 * x**2 - 4 * x + 5

# Create x values for plotting
x = np.linspace(-2, 4, 100)
y = f(x)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', linewidth=2, label='f(x) = 3x² - 4x + 5')
plt.grid(True, alpha=0.3)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Plot of f(x) = 3x² - 4x + 5')
plt.legend()

# Add some special points that were evaluated in the Lean file
eval_points_x = [0, 1, 3]
eval_points_y = [f(x) for x in eval_points_x]
plt.scatter(eval_points_x, eval_points_y, color='red', s=50, zorder=5, 
           label='Evaluated points')

# Add annotations for the evaluated points
for x_val, y_val in zip(eval_points_x, eval_points_y):
    plt.annotate(f'({x_val}, {y_val})', 
                xy=(x_val, y_val), 
                xytext=(5, 5), 
                textcoords='offset points',
                fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

# Find and mark the vertex of the parabola
vertex_x = 4 / (2 * 3)  # -b / (2a) where a=3, b=-4
vertex_y = f(vertex_x)
plt.scatter([vertex_x], [vertex_y], color='green', s=100, marker='*', 
           zorder=5, label=f'Vertex ({vertex_x:.2f}, {vertex_y:.2f})')

plt.legend()
plt.tight_layout()
plt.show()

# Print some information about the function
print("Function: f(x) = 3x² - 4x + 5")
print(f"Vertex: ({vertex_x:.3f}, {vertex_y:.3f})")
print("Evaluated points from Lean:")
for x_val in eval_points_x:
    print(f"  f({x_val}) = {f(x_val)}")