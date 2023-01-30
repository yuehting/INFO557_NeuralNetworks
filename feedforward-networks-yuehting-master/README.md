# Objectives

The learning objectives of this assignment are to:
1. get familiar with the TensorFlow Keras framework for training neural networks.
2. experiment with the various hyper-parameter choices of feedforward networks.

# Setup your environment

You will need to set up an appropriate coding environment on whatever computer
you expect to use for this assignment.
Minimally, you should install:

* [git](https://git-scm.com/downloads)
* [Python (version 3.8 or higher)](https://www.python.org/downloads/)
* [tensorflow (version 2.9)](https://www.tensorflow.org/)
* [pytest](https://docs.pytest.org/)

# Check out the starter code

After accepting the assignment on GitHub Classroom, clone the newly created
repository to your local machine:
```
git clone https://github.com/ua-ista-457/feedforward-networks-<your-username>.git
```
You are now ready to begin working on the assignment.
You should do all your work in the default branch, `main`.

# Write your code

You will implement several feedforward neural networks using the
[TensorFlow Keras API](https://www.tensorflow.org/guide/keras/).
You should read the documentation strings (docstrings) in each of methods in
`nn.py`, and implement the methods as described.
Write your code below the docstring of each method;
**do not delete the docstrings**.

The following objects and functions may come in handy:
* [Sequential](https://www.tensorflow.org/api_docs/python/tf/keras/Sequential)
* [Sequential.compile](https://www.tensorflow.org/api_docs/python/tf/keras/Sequential#compile)
* [Dense](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense)
* [Dropout](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dropout)
* [EarlyStopping](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping)

# Test your code for correctness

Tests have been provided for you in the `test_nn.py` file.
The tests show how each of the methods is expected to be used.
To run all the provided tests, run the ``pytest`` script from the directory
containing ``test_nn.py``.
Initially, you will see output like:
```
============================= test session starts ==============================
...
collected 4 items

test_nn.py FFFF                                                          [100%]

=================================== FAILURES ===================================
...
============================== 4 failed in 7.33s ===============================
```
This indicates that all tests are failing, which is expected since you have not
yet written the code for any of the methods.
Once you have written the code for all methods, you should instead see
something like:
```
============================= test session starts ==============================
...
collected 4 items

test_nn.py
8.2 RMSE for baseline on Auto MPG
6.2 RMSE for deep on Auto MPG
3.9 RMSE for wide on Auto MPG
.
65.0% accuracy for baseline on del.icio.us
68.7% accuracy for relu on del.icio.us
66.9% accuracy for tanh on del.icio.us
.
18.2% accuracy for baseline on UCI-HAR
93.8% accuracy for dropout on UCI-HAR
91.7% accuracy for no dropout on UCI-HAR
.
75.4% accuracy for baseline on census income
79.0% accuracy for early on census income
77.8% accuracy for late on census income
.                                                          [100%]

============================== 4 passed in 23.16s ==============================
```
**Warning**: The performance of your models may change somewhat from run to run,
especially when moving from one machine to another, since neural network models
are randomly initialized.
A correct solution to this assignment should pass the tests on any machine.
Make sure that the tests are passing on GitHub!
If you see that they are failing on GitHub even though they are passing on your
local machine, you will likely need to change your code.
Read the build log on GitHub to see if you have any coding errors;
otherwise, try different hyper-parameters for your model.

# Test your code for quality

In addition to the correctness tests, you should run `pylint nn.py` to check
for common code quality problems.
Pylint will check for adherence to
[standard Python style](https://www.python.org/dev/peps/pep-0008/),
good variable names, proper use of builtins like `enumerate`, etc.
Depending on how you import from Tensorflow/Keras, you may need to specify the
` --ignored-modules=tensorflow --disable=no-name-in-module` options to silence
lint warnings from those packages.
If there are no problems, you should see something like:
```
--------------------------------------------------------------------
Your code has been rated at 10.00/10 (previous run: 10.00/10, +0.00)
```

# Submit your code

As you are working on the code, you should regularly `git commit` to save your
current changes locally.
You should also regularly `git push` to push all saved changes to the remote
repository on GitHub.
Make a habit of checking the "Feedback" pull request on the GitHub page for your
repository.
You should see all your pushed commits there, as well as the status of the
"checks".
If any correctness (pytest) or quality (pylint) tests are failing, you will see
"All checks have failed" at the bottom of the pull request.
If you want to see exactly which tests have failed, click on the "Details" link.
When you have corrected all problems, you should see "All checks have passed"
at the bottom of the pull request.

You do not need to do anything beyond pushing your commits to Github to submit
your assignment.
The instructional team will grade the code of the "Feedback" pull request, and
make detailed comments there.

# Grading

The points are allocated as follows:
* 80 points for passing all automated correctness (pytest) tests
* 10 points for passing all automated quality (pylint) tests
* 10 points for other quality issues:
using appropriate data structures,
using existing library functions whenever appropriate,
minimizing code duplication,
giving variables meaningful names,
documenting complex pieces of code, etc.
