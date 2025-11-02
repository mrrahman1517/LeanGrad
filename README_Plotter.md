# 📊 Parameterized Lean Function Plotter

An advanced automated system for plotting mathematical functions defined in Lean 4, featuring **parameterized function passing** for maximum flexibility and modularity.

## 🚀 Quick Start

### Method 1: Interactive Parameterized Menu
```bash
python parameterized_plotter.py
```
Choose from multiple predefined functions with different parameters.

### Method 2: Original Direct Plotting
```bash
python auto_plot.py
```
Plots whatever function is currently active in `DataGenerator.lean`.

### Method 3: Multi-Function Demo
```bash
lean MultiFunctionExample.lean && python auto_plot.py
```
See parameterized functions with different configurations.

## ✨ New Parameterized System

### Key Innovation: Functions as Parameters!

Instead of hardcoding functions, the new system passes functions as parameters:

```lean
-- OLD WAY (fixed function):
def f(x : Float): Float := 3 * x^2 - 4 * x + 5
#eval generateData -- always uses f

-- NEW WAY (function as parameter):
def quadratic (x : Float) : Float := 3 * x^2 - 4 * x + 5
def sine (x : Float) : Float := Float.sin x

-- Plot any function with custom parameters:
#eval plotFunction quadratic "f" "3x² - 4x + 5" (-2.0) 4.0 100 [0, 1, 3]
#eval plotFunction sine "g" "sin(x)" (-6.28) 6.28 200 [0, 1.57, 3.14]
```

### Benefits:
- 🔄 **Switch functions instantly** without editing code
- ⚙️ **Custom parameters** for each function (range, resolution, special points)
- 📚 **Function library** with reusable definitions
- 🎯 **Type safety** - functions are properly typed parameters

## 📝 Using the Parameterized System

### Step 1: Define Functions
```lean
-- In DataGenerator.lean or your own file:
def myQuadratic (x : Float) : Float := 2 * x^2 - x + 3
def mySine (x : Float) : Float := 3 * Float.sin (2 * x)
def myCustom (x : Float) : Float := x * Float.exp (-x^2)
```

### Step 2: Plot with Parameters
```lean
-- Plot quadratic: function, name, expression, start, stop, points, special_x_values
#eval plotFunction myQuadratic "q" "2x² - x + 3" (-3.0) 3.0 150 [-1, 0, 1, 2]

-- Plot sine wave with different parameters
#eval plotFunction mySine "s" "3sin(2x)" (-3.14) 3.14 200 [0, 0.785, 1.57]

-- Plot custom function
#eval plotFunction myCustom "c" "x·e^(-x²)" (-3.0) 3.0 300 [-2, -1, 0, 1, 2]
```

### Step 3: Run the Plotter
```bash
python auto_plot.py
```

## 🎨 Parameterized Function Library

### Polynomial Functions with Parameters
```lean
def linear (a b : Float) (x : Float) : Float := a * x + b
def quadratic (a b c : Float) (x : Float) : Float := a * x^2 + b * x + c
def cubic (a b c d : Float) (x : Float) : Float := a * x^3 + b * x^2 + c * x + d

-- Usage:
#eval plotFunction (quadratic 2 (-3) 1) "f" "2x² - 3x + 1" (-2.0) 4.0 100 [0, 0.75, 2]
```

### Trigonometric Functions with Parameters
```lean
def sine (amplitude frequency phase : Float) (x : Float) : Float := 
  amplitude * Float.sin (frequency * x + phase)

def cosine (amplitude frequency phase : Float) (x : Float) : Float := 
  amplitude * Float.cos (frequency * x + phase)

-- Usage:
#eval plotFunction (sine 2 3 0) "g" "2sin(3x)" (-2.1) 2.1 200 [0, 1.047, 2.094]
```

### Advanced Parameterized Functions
```lean
def gaussian (amplitude sigma mu : Float) (x : Float) : Float := 
  amplitude * Float.exp (-(x - mu)^2 / (2 * sigma^2))

def dampedOscillation (decay frequency : Float) (x : Float) : Float := 
  Float.exp (-decay * x) * Float.sin (frequency * x)

-- Usage:
#eval plotFunction (gaussian 1 1 0) "bell" "e^(-x²/2)" (-4.0) 4.0 200 [-2, 0, 2]
#eval plotFunction (dampedOscillation 0.1 2) "damped" "e^(-0.1x)sin(2x)" 0 20 300 [0, 1.57, 3.14]
```

## 📁 Enhanced File Structure

```
micrograd/
├── DataGenerator.lean           # 🔧 Main parameterized system
├── MultiFunctionExample.lean    # 📚 Multiple function examples
├── FunctionTemplates.lean       # 📝 Template library
├── SineExample.lean            # 🌊 Pre-configured sine example
├── auto_plot.py               # 🎨 Enhanced plotting engine
├── parameterized_plotter.py   # 🚀 Interactive function selector
├── quick_plot.py              # ⚡ Quick function switcher (legacy)
└── README_Plotter.md          # 📖 This documentation
```

## 🔧 Advanced Parameterization Examples

### Example 1: Family of Parabolas
```lean
-- Define a family of quadratic functions
def parabolaFamily (a : Float) (x : Float) : Float := a * x^2

-- Plot different family members:
-- #eval plotFunction (parabolaFamily 1) "f₁" "x²" (-3.0) 3.0 100 [-2, 0, 2]
-- #eval plotFunction (parabolaFamily 2) "f₂" "2x²" (-3.0) 3.0 100 [-2, 0, 2]
-- #eval plotFunction (parabolaFamily 0.5) "f₃" "½x²" (-3.0) 3.0 100 [-2, 0, 2]
```

### Example 2: Transformations
```lean
-- Function transformations as parameters
def transform (base : Float → Float) (stretch scale shift : Float) (x : Float) : Float :=
  stretch * base (scale * (x - shift))

def baseSine (x : Float) : Float := Float.sin x

-- Apply transformations:
-- #eval plotFunction (transform baseSine 2 1 0) "t₁" "2sin(x)" (-6.28) 6.28 200 []
-- #eval plotFunction (transform baseSine 1 2 0) "t₂" "sin(2x)" (-6.28) 6.28 200 []
-- #eval plotFunction (transform baseSine 1 1 1.57) "t₃" "sin(x - π/2)" (-6.28) 6.28 200 []
```

### Example 3: Composition of Functions
```lean
-- Function composition as parameters
def compose (f g : Float → Float) (x : Float) : Float := f (g x)

def square (x : Float) : Float := x^2
def sine (x : Float) : Float := Float.sin x

-- Compose functions:
-- #eval plotFunction (compose square sine) "comp₁" "sin²(x)" (-6.28) 6.28 200 []
-- #eval plotFunction (compose sine square) "comp₂" "sin(x²)" (-3.0) 3.0 200 []
```

## 🎯 Best Practices for Parameterized Functions

1. **Meaningful Names**: Use descriptive function names like `dampedSine` not `f1`
2. **Parameter Documentation**: Comment your parameter meanings
3. **Appropriate Ranges**: Choose domains that show interesting behavior
4. **Special Points**: Select meaningful x-values (zeros, extrema, inflection points)
5. **Resolution**: Use more points (200-500) for smooth curves

## 💡 Tips for Advanced Usage

### Creating Function Families
```lean
-- Create functions that return functions (higher-order)
def makePolynomial (coeffs : List Float) : Float → Float :=
  fun x => coeffs.enum.foldl (fun acc (i, c) => acc + c * x^i.toFloat) 0

-- Usage:
def cubic := makePolynomial [1, -2, 0, 1]  -- 1 - 2x + x³
-- #eval plotFunction cubic "p" "1 - 2x + x³" (-2.0) 2.0 150 [-1, 0, 1]
```

### Conditional/Piecewise Functions
```lean
def piecewise (x : Float) : Float :=
  if x < 0 then -x^2
  else if x < 2 then x
  else (x - 2)^2 + 2

-- #eval plotFunction piecewise "pw" "piecewise function" (-3.0) 4.0 200 [-2, 0, 2, 3]
```

### Function Debugging
```lean
-- Test function values before plotting
#eval myFunction 0    -- Test at specific points
#eval myFunction 1
#eval myFunction (-1)

-- Then plot:
#eval plotFunction myFunction "test" "my function" (-2.0) 2.0 100 [0, 1, -1]
```

Ready to explore the power of parameterized mathematical functions! 🚀

The new system gives you the flexibility of functional programming with the precision of Lean and beautiful Python visualizations.