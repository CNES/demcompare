# Scalar metrics:

`demcompare` computes scalar metrics:
- On a classified group of pixels 
- Gives a float or an int as a result
- Can handle parameters 
- Each metric is a subclass of `MetricTemplate`

# Existing tests:

### Tests ScalarMetric class (tests/metric/test_scalar_metric.py)

There is a unit test by metric, they all have the same objective : 

**Test objective**

Test the metric class function `compute_metric`.
    Manually computes an input array and
    computes its metric,
    and tests that the resulting
    arrays form the `metric_class.compute_metric` function are the
    same.


### `test_mean`

### `test_max`

### `test_min`

### `test_std`

### `test_rmse`

### `test_median`

### `test_nmad`

### `test_sum`

### `test_percentil_90`
