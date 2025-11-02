import Micrograd.Basic

def f(x : Float): Float := 3 * x^2 - 4 * x + 5

-- Generate a list of x values
def xValues : List Float := [-2, -1, 0, 1, 2, 3, 4]

-- Generate corresponding y values
def yValues : List Float := xValues.map f

-- Display the data points
#eval xValues.zip yValues

-- Or generate more points programmatically
def generatePoints (start stop : Float) (n : Nat) : List (Float × Float) :=
  let step := (stop - start) / n.toFloat
  let xVals := List.range n |>.map (fun i => start + i.toFloat * step)
  xVals.map (fun x => (x, f x))

#eval generatePoints (-2) 4 10