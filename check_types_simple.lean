-- Check available numeric types in Lean 4

#check Float

-- Try to see what Float actually is
#print Float

-- Check precision
def testFloat : Float := 1.0 / 3.0
#eval testFloat

-- Check what's available
#check Nat
#check Int

-- Check if there are specific sized floats
-- #check Float32
-- #check Float64

-- Let's see if there's a Rational type available in core
-- #check Rat
