# v0.1.0

## Interface changes

### Optimizer
Switch from `ipopt` to `scipy.optimize.minimize`, the original optimization options are deprecated. The deprecated arguments in `fit_model` function for `MRBRT` and `MRBeRT` include,
  * `inner_print_level`
  * `inner_max_iter`
  * `inner_tol`

For the new arguments please refer to limetr v0.0.5.

### Knots sampling
For knots sampling, we moved away from pycddlib due to its installation issues in certain operating systems, instead we use a simple algorithm to create knots.
* In the function `utils.sample_knots` `interval_sizes` argument is depercated and replaced by `min_dist`. The original `interval_sizes` corresponding to the lower and upper bounds of the size of the interval and `min_dist` only refers to the lower bounds of the size of the interval.
* `model.create_knots_samples` has been removed.

