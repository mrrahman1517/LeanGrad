-- ========================================
-- DEFINE YOUR FUNCTION HERE
-- ========================================
-- Example: Sine wave function
def f(x : Float): Float := Float.sin x

-- ========================================
-- CONFIGURE PLOTTING PARAMETERS
-- ========================================
-- Adjust these to fit your function's domain and desired resolution
def plotStart : Float := -6.28    -- Start of x range (-2π)
def plotStop : Float := 6.28      -- End of x range (2π)
def plotPoints : Nat := 200       -- Number of points to generate (higher = smoother curve)

-- Define special points to evaluate and highlight
-- These will be marked with red dots on the plot
def specialXValues : List Float := [0, 1.57, 3.14, 4.71]  -- 0, π/2, π, 3π/2

-- ========================================
-- AUTOMATIC FUNCTION METADATA EXTRACTION
-- ========================================
-- Try to automatically determine function properties
def functionName : String := "f"
def functionExpression : String := "sin(x)"

-- ========================================
-- PLOTTING LOGIC (NO NEED TO MODIFY)
-- ========================================

-- Helper function to generate evenly spaced points
def linspace (start stop : Float) (n : Nat) : List Float :=
  if n = 0 then []
  else if n = 1 then [start]
  else
    let step := (stop - start) / (n - 1).toFloat
    List.range n |>.map (fun i => start + i.toFloat * step)

-- Generate data points for plotting
def generatePlotData (start stop : Float) (n : Nat) : List (Float × Float) :=
  let xVals := linspace start stop n
  xVals.map (fun x => (x, f x))

-- Generate the data using configured parameters
def plotData : List (Float × Float) := generatePlotData plotStart plotStop plotPoints

-- Output data in a format Python can easily parse
def formatPoint (p : Float × Float) : String :=
  s!"{p.1},{p.2}"

def exportData : String :=
  "x,y\n" ++ (plotData.map formatPoint |>.foldl (fun acc s => acc ++ s ++ "\n") "")

-- Special points to highlight (using configured values)
def specialPoints : List (Float × Float) := specialXValues.map (fun x => (x, f x))

def exportSpecialPoints : String :=
  "x,y\n" ++ (specialPoints.map formatPoint |>.foldl (fun acc s => acc ++ s ++ "\n") "")

-- Generate metadata for the Python plotter
def functionMetadata : String :=
  s!"function_name,{functionName}\nfunction_expr,{functionExpression}\nplot_start,{plotStart}\nplot_stop,{plotStop}\nplot_points,{plotPoints}"

-- Write data to files when this is run
#eval IO.FS.writeFile "plot_data.csv" exportData
#eval IO.FS.writeFile "special_points.csv" exportSpecialPoints
#eval IO.FS.writeFile "function_metadata.csv" functionMetadata

-- Also display some info
#eval s!"Function: {functionName}(x) = {functionExpression}"
#eval s!"Generated {plotData.length} data points"
#eval s!"Range: {plotStart} to {plotStop}"
#eval s!"Special points: {specialPoints}"
