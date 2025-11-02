# Automated Lean + Python Plotting Workflow

.PHONY: plot clean help

# Default target
plot:
	@echo "🚀 Running automated Lean → Python plotting workflow..."
	python auto_plot.py

# Clean generated files
clean:
	@echo "🧹 Cleaning generated files..."
	rm -f plot_data.csv special_points.csv
	@echo "✅ Cleanup complete"

# Generate data only (without plotting)
data:
	@echo "📊 Generating data with Lean..."
	lean DataGenerator.lean
	@echo "✅ Data generation complete"

# Plot only (using existing data)
plot-only:
	@echo "📈 Creating plot from existing data..."
	@python -c "\
import pandas as pd; \
import matplotlib.pyplot as plt; \
import numpy as np; \
exec(open('auto_plot.py').read()); \
plot_data, special_data = load_data(); \
create_plot(plot_data, special_data) if plot_data is not None and special_data is not None else print('❌ No data found. Run make data first.')"

# Show help
help:
	@echo "📋 Available commands:"
	@echo "  make plot     - Generate data with Lean and plot with Python (default)"
	@echo "  make data     - Generate data with Lean only"
	@echo "  make plot-only- Plot using existing data"
	@echo "  make clean    - Remove generated CSV files"
	@echo "  make help     - Show this help message"
	@echo ""
	@echo "🔧 To modify the function:"
	@echo "  1. Edit the function 'f' in DataGenerator.lean"
	@echo "  2. Run 'make plot' to see the updated plot"