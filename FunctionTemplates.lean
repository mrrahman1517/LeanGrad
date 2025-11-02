-- ========================================
-- FUNCTION TEMPLATE EXAMPLES
-- ========================================
-- Copy any of these function definitions to DataGenerator.lean
-- and adjust the parameters as needed!

-- POLYNOMIAL FUNCTIONS
-- def f(x : Float): Float := x^2                           -- Simple parabola
-- def f(x : Float): Float := -x^2 + 4*x - 3                -- Downward parabola
-- def f(x : Float): Float := x^3 - 3*x^2 + 2*x + 1         -- Cubic function
-- def f(x : Float): Float := x^4 - 4*x^2 + 3               -- Quartic function

-- TRIGONOMETRIC FUNCTIONS
-- def f(x : Float): Float := Float.sin x                   -- Sine wave
-- def f(x : Float): Float := Float.cos x                   -- Cosine wave
-- def f(x : Float): Float := Float.tan x                   -- Tangent (be careful with domain!)
-- def f(x : Float): Float := 2 * Float.sin (3 * x) + 1    -- Scaled/shifted sine

-- EXPONENTIAL/LOGARITHMIC FUNCTIONS
-- def f(x : Float): Float := Float.exp x                   -- e^x
-- def f(x : Float): Float := 2^x                           -- 2^x
-- def f(x : Float): Float := Float.log x                   -- ln(x) (x > 0 only!)
-- def f(x : Float): Float := Float.exp (-x^2)              -- Gaussian bell curve

-- RATIONAL FUNCTIONS
-- def f(x : Float): Float := 1/x                           -- Hyperbola (x ≠ 0)
-- def f(x : Float): Float := (x^2 + 1)/(x - 2)             -- More complex rational
-- def f(x : Float): Float := x/(x^2 + 1)                   -- Bounded rational

-- PIECEWISE/CONDITIONAL FUNCTIONS
-- def f(x : Float): Float := if x < 0 then -x else x^2     -- Piecewise function
-- def f(x : Float): Float := if x.abs < 1 then 1 else 0    -- Step-like function

-- COMBINATION FUNCTIONS
-- def f(x : Float): Float := x * Float.sin x               -- x * sin(x)
-- def f(x : Float): Float := Float.exp x * Float.cos x     -- e^x * cos(x)
-- def f(x : Float): Float := Float.sqrt (x.abs)            -- √|x|

-- ========================================
-- SUGGESTED PLOTTING RANGES
-- ========================================
-- Polynomial functions: [-5, 5] usually works well
-- Trigonometric: [-2π, 2π] or [-6.28, 6.28]
-- Exponential: [-3, 3] for exp(x), [0.1, 5] for log(x)
-- Rational: Be careful around discontinuities!
--
-- Example configurations:
--
-- For trigonometric functions:
-- def plotStart : Float := -6.28    -- -2π
-- def plotStop : Float := 6.28      -- 2π
-- def plotPoints : Nat := 200       -- More points for smooth curves
--
-- For rational functions like 1/x:
-- def plotStart : Float := -5.0
-- def plotStop : Float := 5.0
-- def plotPoints : Nat := 200
-- def specialXValues : List Float := [-2, -1, 1, 2]  -- Avoid x=0

-- ========================================
-- USAGE INSTRUCTIONS
-- ========================================
-- 1. Copy a function definition above to DataGenerator.lean (replace the existing f)
-- 2. Update functionExpression string to match your function
-- 3. Adjust plotStart, plotStop, plotPoints for your function's domain
-- 4. Set specialXValues to interesting points you want to highlight
-- 5. Run: python auto_plot.py
