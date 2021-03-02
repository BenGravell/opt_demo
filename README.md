# opt_demo
Demo of optimization algorithms for infinite-horizon discrete-time LQR policy optimization

## Dependencies
* [NumPy](https://github.com/numpy/numpy)
* [SciPy](https://github.com/scipy/scipy)
* [Autograd](https://github.com/HIPS/autograd)
* [matplotlib](https://github.com/matplotlib/matplotlib)

## File structure & capabilities

### `opt_demo.py`
Main script.
* Initialize settings
* Call optimizers
* Aggregate results
* Plot summary metrics to show convergence

### `optimizers.py`
Core code for object-oriented generic unconstrained optimization methods.
* Accelerated gradient methods
  * Gradient descent w/ constant step size
  * Polyak momentum
  * Nesterov acceleration
  * Relativistic acceleration 
  * RMSprop
  * Adam
* Line search methods (backtracking and Wolfe)
  * Gradient
  * Newton w/ Hessian modification
  * Quasi-Newton
    * BFGS
    * DFP
    * SR1 
* Trust region methods
  * Dogleg
  * 2D-subspace minimization
  * Conjugate gradient (Steihaug)

Additionally, the code includes a thin wrapper around the [unconstrained optimization algorithms implemented by SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize).
* Nelder-Mead
* Powell
* CG
* Newton-CG
* BFGS
* dogleg
* trust-ncg
* trust-exact
* trust-krylov

### `lqr_optimizers.py`
Dynamic programming solutions for the infinite-horizon discrete-time LQR problem.
* Policy iteration
* Value iteration

### `lqr_problems.py`
Functions to generate random problem data and create the objective function, its gradient, and its Hessian.

### `lqr_utility.py`
Linear algebraic functions needed to define and work with infinite-horizon discrete-time LQR policy optimization.
