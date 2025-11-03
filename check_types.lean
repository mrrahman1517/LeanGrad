-- Check available numeric types in Lean 4
import Mathlib.Data.Real.Basic

#check Float

-- Try to see what Float actually is
#print Float

-- Check precision
def testFloat : Float := 1.0 / 3.0
#eval testFloat

-- Check what's available
#check Nat
#check Int
#check Rat

-- Try importing Math library for Real numbers
#check Real
def testReal : Real := 1.0 / 3.0
