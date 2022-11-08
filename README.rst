.. image:: https://img.shields.io/badge/License-BSD%202--Clause-orange.svg
    :target: https://opensource.org/licenses/BSD-2-Clause
    :alt: License

.. image:: https://readthedocs.org/projects/mrtool/badge/?version=latest
    :target: https://mrtool.readthedocs.io/en/latest/
    :alt: Documentation

.. image:: https://github.com/ihmeuw-msca/workflows/build/badge.svg?branch=main
    :target: https://github.com/ihmeuw-msca/mrtool/actions
    :alt: BuildStatus

.. image:: https://badge.fury.io/py/mrtool.svg
    :target: https://badge.fury.io/py/mrtool
    :alt: PyPI


MRTool
======


**MRTool** (Meta-Regression Tool) package is designed to solve general meta-regression problem.
The most interesting features include,

* linear and log prediction function,
* spline extension for covariates,
* direct Gaussian, Uniform and Laplace prior on fixed and random effects,
* shape constraints (monotonicity and convexity) for spline.

Advanced features include,

* spline knots ensemble,
* automatic covariate selection.


Installation
------------

Use pip to install the package

.. code-block:: shell

   pip install mrtool


For more information please check the `documentation <https://mrtool.readthedocs.io/en/latest>`_.
