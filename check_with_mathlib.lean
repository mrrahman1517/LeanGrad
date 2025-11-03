import Mathlib.Data.Real.Basic

-- Check available numeric types with mathlib

#check Float
#check Real

-- Compare precision
def testFloat : Float := 1.0 / 3.0
def testReal : Real := (1 : Real) / 3

#eval testFloat

-- Note: Real numbers in Lean are exact but can't be #eval'd directly
-- They're used for proofs and theoretical work

-- Check what other numeric types are available
#check Rat  -- Rational numbers (exact fractions)

-- Try a rational computation
def testRat : Rat := (1 : Rat) / 3
#eval testRat
