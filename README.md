# LeanGrad

A formally verified autograd engine implemented in Lean 4, featuring parameterized function plotting and type-safe gradient computation.

## 🚀 Overview

LeanGrad combines the mathematical rigor of Lean's dependent type system with automatic differentiation capabilities, inspired by Andrej Karpathy's micrograd but built on formal foundations.

## ✨ Features

- **🔧 Parameterized Function System**: Functions passed as parameters for maximum flexibility
- **📊 Advanced Plotting**: Automated visualization with Python integration
- **🎯 Type Safety**: Lean's type system ensures mathematical correctness
- **📚 Function Libraries**: Reusable, parameterized mathematical functions
- **🔄 Modular Design**: Clean separation between computation and visualization

## 📁 Project Structure

```
LeanGrad/
├── Micrograd.lean              # Core autograd implementation
├── Micrograd/Basic.lean        # Basic definitions
├── DataGenerator.lean          # Parameterized plotting system
├── MultiFunctionExample.lean   # Advanced function examples
├── FunctionTemplates.lean      # Function template library
├── auto_plot.py               # Python plotting engine
├── parameterized_plotter.py   # Interactive function plotter
└── README_Plotter.md          # Detailed plotting documentation
```

## 🎯 Quick Start

### Plotting Functions
```bash
# Interactive function plotter
python parameterized_plotter.py

# Direct plotting
python auto_plot.py

# System demonstration
python demo_parameterized.py
```

### Function Definition (Lean)
```lean
-- Define parameterized functions
def quadratic (a b c : Float) (x : Float) : Float := a * x^2 + b * x + c

-- Plot with custom parameters
#eval plotFunction (quadratic 3 (-4) 5) "f" "3x² - 4x + 5" (-2.0) 4.0 100 [0, 1, 3]
```

## 🔬 Research Goals

- Formally verified gradient computations
- Type-safe automatic differentiation
- Integration of theorem proving with machine learning
- Mathematically rigorous neural network foundations

## 📖 Documentation

See [README_Plotter.md](README_Plotter.md) for detailed documentation of the parameterized plotting system.

## 🛠️ Development

Built with Lean 4 and Lake build system. Python components require matplotlib, pandas, and numpy.

---

*LeanGrad: Where formal verification meets automatic differentiation* 🎓⚡
