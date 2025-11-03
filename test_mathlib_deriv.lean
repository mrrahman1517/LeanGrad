-- Test Mathlib differentiation capabilities
import Mathlib.Analysis.Calculus.FDeriv.Basic
import Mathlib.Analysis.Calculus.Deriv.Basic
import Mathlib.Analysis.Calculus.FDeriv.Prod
import Mathlib.Data.Real.Basic

-- Check what's available for differentiation in Mathlib
open Real

-- Define a simple function using Real numbers
def g : ℝ → ℝ := fun x => 3 * x^2 - 4 * x + 5

-- Try to use Mathlib's derivative functions
#check deriv g  -- Mathlib's derivative function
#check fderiv ℝ g  -- Frechet derivative

-- Check some basic derivative theorems
#check deriv_const
#check deriv_id'
#check deriv_add
#check deriv_mul
#check deriv_pow

-- Try to compute derivative symbolically if possible
example : deriv g = fun x => 6 * x - 4 := by
  funext x
  simp [g, deriv_add, deriv_sub, deriv_mul, deriv_const, deriv_pow, deriv_id']
  ring

-- Test evaluation (though Real can't be #eval'd directly)
-- We can prove properties about the derivative
