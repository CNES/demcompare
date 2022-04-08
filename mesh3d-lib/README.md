# CNES R&T Mesh 3D

## Install

The python library package and the CLI entrypoint can be installed with pip:

```shell
$ pip install <path/to/mesh3d-lib>
```

If the library is to be edited on the fly, one can install it with the editable option:
```shell
$ pip install -e <path/to/mesh3d-lib>
```

### Tests and lint

If one wishes to install the dependencies required to run the test
suites and lint the code, the wheels provide extra-requirements groups 
for this very purpose.

The `tests` and `lint` requirement groups will install the necessary
packages.

The test suite runs with pytest. A nox configuration is provided for both cases.
