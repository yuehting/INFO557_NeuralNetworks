# Objectives

The learning objectives of this assignment are to:
1. practice Python programming skills and use of numpy arrays
2. get familiar with submitting assignments on GitHub Classroom

# Setup your environment

You will need to set up an appropriate coding environment on whatever computer
you expect to use for this assignment.
Minimally, you should install:

* [git](https://git-scm.com/downloads)
* [Python (version 3.8 or higher)](https://www.python.org/downloads/)
* [numpy](http://www.numpy.org/)
* [pytest](https://docs.pytest.org/)
* [pytest-timeout](https://pypi.org/project/pytest-timeout/)

If you have not used Git, Python, or Numpy before, this would be a good time to
go through some tutorials:

* [git tutorial](https://try.github.io/)
* [Python tutorial](https://docs.python.org/3/tutorial/)
* [numpy tutorial](https://docs.scipy.org/doc/numpy/user/quickstart.html)

You can find many other tutorials for these tools online.

# Check out the starter code

When you accepted the assignment, Github created a clone of the assignment
template for you at:

```
https://github.com/ua-ista-457/strings-to-vectors-<your-username>
```

It also set up a separate `feedback` branch and a
[feedback pull request](https://docs.github.com/en/education/manage-coursework-with-github-classroom/teach-with-github-classroom/leave-feedback-with-pull-requests)
that will allow the instructional team to give you feedback on your work.

To get the assignment code on your local machine, clone the repository:
```
git clone https://github.com/ua-ista-457/strings-to-vectors-<your-username>.git
```
You are now ready to begin working on the assignment.
You should do all your work in the default branch, `main`.

# Write your code

You will implement an `Index` that associates objects with integer indexes.
This is a very common setup step in training neural networks, which require that
everything be expressed as numbers, not objects.

A template for the `Index` class has been provided to you in the file `nn.py`.
In the template, each method has only a documentation string, with no code in
the body of the method yet.
For example, the `objects_to_indexes` method looks like:
```python
def objects_to_indexes(self, object_seq: Sequence[Any]) -> np.ndarray:
    """
    Returns a vector of the indexes associated with the input objects.

    For objects not in the vocabulary, `start-1` is used as the index.

    :param object_seq: A sequence of objects.
    :return: A 1-dimensional array of the object indexes.
    """
```

You should read the documentation strings (docstrings) in each of methods in
`nn.py`, and implement the methods as described.
Write your code below the docstring of each method;
**do not delete the docstrings**.

The following objects and functions may come in handy:
* [dict](https://docs.python.org/3/library/stdtypes.html#mapping-types-dict)
* [numpy.array](https://numpy.org/doc/stable/reference/generated/numpy.array.html)
* [numpy.full](https://numpy.org/doc/stable/reference/generated/numpy.full.html)
* [numpy.zeros](https://numpy.org/doc/stable/reference/generated/numpy.zeros.html)
* [numpy.stack](https://numpy.org/doc/stable/reference/generated/numpy.stack.html)
* [numpy.nonzero](https://numpy.org/doc/stable/reference/generated/numpy.nonzero.html)

# Test your code for correctness

The `test_nn.py` file contains tests that check that each of the methods of
`Index` behaves as expected.
For example, the `test_indexes` test method looks like:

```python
@pytest.mark.timeout(1)
def test_indexes():
    vocab = ["four", "three", "", "two", "one"]
    objects = ["one", "", "four", "four"]
    indexes = np.array([4, 2, 0, 0])

    index = nn.Index(vocab)
    assert_array_equal(index.objects_to_indexes(objects), indexes)
    assert index.indexes_to_objects(indexes) == objects

    index = nn.Index(vocab, start=1)
    assert_array_equal(index.objects_to_indexes(objects), indexes + 1)
    assert index.indexes_to_objects(indexes + 1) == objects
```
This tests that your code correctly associates indexes with an input vocabulary
``"four", "three", "", "two", "one"``, that it can convert back and forth
between objects and indexes, and that it can handle indexing that starts from a
number other than 0.

To run all the provided tests, run ``pytest`` from the directory containing
``test_nn.py``.
Initially, you will see output like:
```
============================= test session starts ==============================
...
collected 8 items

test_nn.py FFFFFFFF                                                      [100%]

=================================== FAILURES ===================================
_________________________________ test_indexes _________________________________

    @pytest.mark.timeout(1)
    def test_indexes():
        vocab = ["four", "three", "", "two", "one"]
        objects = ["one", "", "four", "four"]
        indexes = np.array([4, 2, 0, 0])

        index = nn.Index(vocab)
>       assert_array_equal(index.objects_to_indexes(objects), indexes)

test_nn.py:21:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

actual = None, desired = array([4, 2, 0, 0])

    def assert_array_equal(actual, desired):
>       assert type(actual) is np.ndarray
E       AssertionError: assert <class 'NoneType'> is <class 'numpy.ndarray'>
E        +  where <class 'NoneType'> = type(None)
E        +  and   <class 'numpy.ndarray'> = np.ndarray

test_nn.py:10: AssertionError
...
============================== 8 failed in 0.73s ===============================
```
This indicates that all tests are failing, which is expected since you have not
yet written the code for any of the methods.
Once you have written the code for all methods, you should instead see
something like:
```
============================= test session starts ==============================
...
collected 8 items

test_nn.py ........                                                      [100%]

============================== 8 passed in 1.49s ===============================
```

# Test your code for quality

In addition to the correctness tests, you should run `pylint nn.py` to check
for common code quality problems.
Pylint will check for adherence to
[standard Python style](https://www.python.org/dev/peps/pep-0008/),
good variable names, proper use of builtins like `enumerate`, etc.
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
