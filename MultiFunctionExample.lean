-- ========================================
-- MULTIPLE FUNCTION PLOTTING EXAMPLE
-- ========================================
-- This file demonstrates how to plot multiple functions easily
-- by using the parameterized data generator

-- Import the plotting infrastructure (if this were a module system)
-- For now, we'll redefine the core functions here

-- Helper function to generate evenly spaced points
def linspace (start stop : Float) (n : Nat) : List Float :=
  if n = 0 then []
  else if n = 1 then [start]
  else
    let step := (stop - start) / (n - 1).toFloat
    List.range n |>.map (fun i => start + i.toFloat * step)

-- Generate data points for plotting (function as parameter)
def generatePlotData (func : Float → Float) (start stop : Float) (n : Nat) : List (Float × Float) :=
  let xVals := linspace start stop n
  xVals.map (fun x => (x, func x))

-- Generate special points (function as parameter)
def generateSpecialPoints (func : Float → Float) (xValues : List Float) : List (Float × Float) :=
  xValues.map (fun x => (x, func x))

-- Format data for CSV export
def formatPoint (p : Float × Float) : String :=
  s!"{p.1},{p.2}"

def formatDataToCsv (data : List (Float × Float)) : String :=
  "x,y\n" ++ (data.map formatPoint |>.foldl (fun acc s => acc ++ s ++ "\n") "")

-- Generate function metadata
def generateMetadata (name expr : String) (start stop : Float) (n : Nat) : String :=
  s!"function_name,{name}\nfunction_expr,{expr}\nplot_start,{start}\nplot_stop,{stop}\nplot_points,{n}"

-- Main plotting function that takes everything as parameters
def plotFunction (func : Float → Float) (name expr : String)
                 (start stop : Float) (n : Nat) (specialX : List Float) : IO Unit := do
  let plotData := generatePlotData func start stop n
  let specialPoints := generateSpecialPoints func specialX
  let metadata := generateMetadata name expr start stop n

  -- Export data
  IO.FS.writeFile "plot_data.csv" (formatDataToCsv plotData)
  IO.FS.writeFile "special_points.csv" (formatDataToCsv specialPoints)
  IO.FS.writeFile "function_metadata.csv" metadata

  -- Display info
  IO.println s!"Function: {name}(x) = {expr}"
  IO.println s!"Generated {plotData.length} data points"
  IO.println s!"Range: {start} to {stop}"
  IO.println s!"Special points: {specialPoints}"

-- ========================================
-- FUNCTION LIBRARY
-- ========================================

-- Polynomial functions
def linear (a b : Float) (x : Float) : Float := a * x + b
def quadratic (a b c : Float) (x : Float) : Float := a * x^2 + b * x + c
def cubic (a b c d : Float) (x : Float) : Float := a * x^3 + b * x^2 + c * x + d

-- Trigonometric functions
def sine (a b c : Float) (x : Float) : Float := a * Float.sin (b * x + c)
def cosine (a b c : Float) (x : Float) : Float := a * Float.cos (b * x + c)

-- Exponential and logarithmic
def exponential (a b : Float) (x : Float) : Float := a * Float.exp (b * x)
def gaussian (a σ μ : Float) (x : Float) : Float := a * Float.exp (-(x - μ)^2 / (2 * σ^2))

-- Rational functions
def rational1 (x : Float) : Float := 1 / x
def rational2 (x : Float) : Float := (x^2 + 1) / (x^2 + 2)

-- Complex combinations
def dampedSine (x : Float) : Float := Float.exp (-0.1 * x) * Float.sin x
def sinTimesX (x : Float) : Float := x * Float.sin x

-- ========================================
-- PLOTTING EXAMPLES
-- ========================================

-- Example 1: Simple quadratic
def f1 := quadratic 1 (-2) 1  -- x² - 2x + 1
-- #eval plotFunction f1 "f₁" "x² - 2x + 1" (-1.0) 3.0 100 [0, 1, 2]

-- Example 2: Scaled sine wave
def f2 := sine 2 3 0  -- 2 sin(3x)
-- #eval plotFunction f2 "f₂" "2sin(3x)" (-3.14) 3.14 200 [0, 1.047, 2.094]

-- Example 3: Gaussian bell curve
def f3 := gaussian 1 1 0  -- e^(-x²/2)
-- #eval plotFunction f3 "f₃" "e^(-x²/2)" (-4.0) 4.0 200 [-2, -1, 0, 1, 2]

-- Example 4: Damped oscillation
-- #eval plotFunction dampedSine "f₄" "e^(-0.1x)·sin(x)" 0 20 300 [0, 3.14, 6.28, 9.42]

-- Example 5: x times sine
-- #eval plotFunction sinTimesX "f₅" "x·sin(x)" (-15.0) 15.0 400 [-10, -5, 0, 5, 10]

-- ========================================
-- ACTIVE FUNCTION (uncomment one to plot)
-- ========================================

-- Choose which function to plot by uncommenting one line:

-- Quadratic function
#eval plotFunction (quadratic 3 (-4) 5) "f" "3x² - 4x + 5" (-2.0) 4.0 100 [0, 1, 3]

-- Sine wave
-- #eval plotFunction (sine 1 1 0) "g" "sin(x)" (-6.28) 6.28 200 [0, 1.57, 3.14, 4.71]

-- Gaussian
-- #eval plotFunction (gaussian 1 1 0) "h" "e^(-x²/2)" (-4.0) 4.0 150 [-2, -1, 0, 1, 2]

-- Damped sine
-- #eval plotFunction dampedSine "i" "e^(-0.1x)·sin(x)" 0 20 300 [0, 3.14, 6.28, 9.42]
