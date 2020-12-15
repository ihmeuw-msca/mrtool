======
MRTool
======

.. image:: https://img.shields.io/badge/License-BSD%202--Clause-orange.svg
    :target: https://opensource.org/licenses/BSD-2-Clause
.. image:: https://readthedocs.org/projects/mrtool/badge/?version=latest
    :target: https://mrtool.readthedocs.io/en/latest/
.. image:: https://travis-ci.com/ihmeuw-msca/MRTool.svg?branch=master
    :target: https://travis-ci.com/github/ihmeuw-msca/MRTool
.. image:: https://badge.fury.io/py/mrtool.svg
    :target: https://badge.fury.io/py/mrtool
.. image:: https://coveralls.io/repos/github/ihmeuw-msca/MRTool/badge.svg?branch=master
    :target: https://coveralls.io/github/ihmeuw-msca/MRTool?branch=master
.. image:: https://scrutinizer-ci.com/g/ihmeuw-msca/MRTool/badges/quality-score.png?b=master
    :target: https://scrutinizer-ci.com/g/ihmeuw-msca/MRTool/?branch=master


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

Required packages include,

* basic scientific computing suite, Numpy, Scipy and Pandas,
* main optimization engine, `IPOPT <https://github.com/matthias-k/cyipopt>`_,
* customized packages, `LimeTr <https://github.com/zhengp0/limetr>`_ and
  `XSpline <https://github.com/zhengp0/xspline>`_,
* testing tool, Pytest.

After install the required packages, clone the repository and install MRTool.

.. code-block:: shell

   git clone https://github.com/ihmeuw-msca/MRTool.git
   cd MRTool && python setup.py install


For more information please check the `documentation <https://mrtool.readthedocs.io/en/latest>`_.

