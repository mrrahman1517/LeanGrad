-- ========================================
-- MODULAR DATA GENERATOR FOR PLOTTING
-- ========================================
-- This system takes functions as parameters for maximum flexibility

-- ========================================
-- CORE PLOTTING INFRASTRUCTURE
-- ========================================

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
-- FUNCTION DEFINITIONS
-- ========================================
-- Define your functions here - they are now completely separate from the plotting logic!

-- Example functions
def quadratic (x : Float) : Float := 3 * x^2 - 4 * x + 5
def cubic (x : Float) : Float := x^3 - 3*x^2 + 2*x + 1
def sine (x : Float) : Float := Float.sin x
def cosine (x : Float) : Float := Float.cos x
def exponential (x : Float) : Float := Float.exp x
def gaussian (x : Float) : Float := Float.exp (-x^2)
def rational (x : Float) : Float := (x^2 + 1) / (x^2 + 2)

-- ========================================
-- USAGE EXAMPLES
-- ========================================
-- Now you can plot any function by calling plotFunction with parameters!

-- Example 1: Plot the quadratic function
#eval plotFunction quadratic "f" "3x² - 4x + 5" (-2.0) 4.0 100 [0, 1, 3]

-- Example 2: Plot a sine wave (uncomment to use)
-- #eval plotFunction sine "g" "sin(x)" (-6.28) 6.28 200 [0, 1.57, 3.14, 4.71]

-- Example 3: Plot an exponential function (uncomment to use)
-- #eval plotFunction exponential "h" "eˣ" (-2.0) 3.0 150 [-1, 0, 1, 2]

-- Example 4: Plot a cubic function (uncomment to use)
-- #eval plotFunction cubic "p" "x³ - 3x² + 2x + 1" (-2.0) 4.0 200 [-1, 0, 1, 2, 3]

-- ========================================
-- CUSTOM FUNCTIONS
-- ========================================
-- You can also define custom functions inline:

-- Custom function example
def myCustomFunction (x : Float) : Float := x * Float.sin x

-- Plot the custom function (uncomment to use)
-- #eval plotFunction myCustomFunction "custom" "x·sin(x)" (-10.0) 10.0 300 [-3.14, 0, 3.14, 6.28]
