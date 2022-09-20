# Vector metrics:

`demcompare` computes vector metrics:
- On a classified group of pixels 
- Gives a `(np.ndarray, np.ndarray)` or a `(List[Union[int, float]], List[Union[int, float]])` as a result
- Can handle parameters 
- Each metric is a subclass of `MetricTemplate`
- Can save the output to a csv or plot file

# Existing tests:

### Tests VectorMetric class (tests/metric/test_vector_metric.py)

There is a unit test by metric, they all have the same objective : 

### `test_ratio_above_threshold`

**Test objective**

Test the `ratio_above_threshold metric` class function
    `compute_metric`.
    Manually computes an input array and
    computes its metric,
    and tests that the resulting
    arrays form the `metric_class.compute_metric` function are the
    same.

Test configuration:
- Test with default elevation threshold
- Test with custom elevation threshold

### `test_cdf`

**Test objective**

Test the `cdf` metric class function
    `compute_metric`.
    Manually computes an input array and
    computes its metric,
    and tests that the resulting
    arrays form the `metric_class.compute_metric` function are the
    same.


### `test_pdf`

**Test objective**

Test the `pdf` metric class function
    `compute_metric`.
    Manually computes an input array and
    computes its metric,
    and tests that the resulting
    arrays form the `metric_class.compute_metric` function are the
    same.

Test configuration:
- Test without percentil 98 filtering
- Test with percentil 98 filtering