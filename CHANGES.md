# v0.1.0

## Interface changes

### Optimizer
Switch from `ipopt` to `scipy.optimize.minimize`, the original optimization options are deprecated. The deprecated arguments in `fit_model` function for `MRBRT` and `MRBeRT` include,
  * `inner_print_level`
  * `inner_max_iter`
  * `inner_tol`
For the new arguments please refer to limetr v0.0.5.
