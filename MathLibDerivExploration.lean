import Mathlib.Analysis.Calculus.Deriv.Basic
import Mathlib.Data.Real.Basic

-- ========================================
-- MATHLIB DIFFERENTIATION EXPLORATION
-- ========================================

-- Mathlib has comprehensive differentiation support!
open Real

-- Define a function using Real numbers (for proofs/theory)
def g : ℝ → ℝ := fun x => 3 * x^2 - 4 * x + 5

-- Mathlib's built-in derivative function
#check deriv g

-- Available differentiation rules in Mathlib:
#check deriv_const     -- d/dx(c) = 0
#check deriv_id'       -- d/dx(x) = 1
#check deriv_add       -- d/dx(f + g) = f' + g'
#check deriv_sub       -- d/dx(f - g) = f' - g'
#check deriv_mul       -- d/dx(f * g) = f' * g + f * g' (product rule)
#check deriv_div       -- quotient rule
#check deriv_pow       -- d/dx(x^n) = n * x^(n-1)
#check deriv_sin       -- d/dx(sin x) = cos x
#check deriv_cos       -- d/dx(cos x) = -sin x
#check deriv_exp       -- d/dx(exp x) = exp x
#check deriv_log       -- d/dx(log x) = 1/x

-- Prove that our derivative is correct using Mathlib
theorem g_derivative : deriv g = fun x => 6 * x - 4 := by
  funext x
  simp only [g, deriv_add, deriv_sub, deriv_mul, deriv_const, deriv_pow, deriv_id']
  ring

-- Chain rule example
def h : ℝ → ℝ := fun x => (3 * x^2 - 4 * x + 5)^2
#check deriv h

-- Mathlib can prove complex derivative properties
theorem chain_rule_example :
  deriv h = fun x => 2 * (3 * x^2 - 4 * x + 5) * (6 * x - 4) := by
  funext x
  simp only [h, deriv_pow, g]
  simp only [deriv_add, deriv_sub, deriv_mul, deriv_const, deriv_pow, deriv_id']
  ring
