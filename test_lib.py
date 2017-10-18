import pytest
from lib import *


def test_1d_constraint():
    x = Variable('x')
    polytope = (x >= 3) & (x <= 5)
    solution = polytope.maximize(x)
    assert solution[x] == pytest.approx(5)
    solution = polytope.minimize(x)
    assert solution[x] == pytest.approx(3)


def test_1d_equality():
    x = Variable('x')
    polytope = (x == 7)
    solution = polytope.maximize(x)
    assert solution[x] == pytest.approx(7)


def test_simple():
    x = Variable('x', lo=0)
    y = Variable('y', lo=0)

    polytope = (x + 3*y <= 10) & \
               (3*x + y <= 10)

    solution = polytope.maximize(x+y)
    assert solution[x] == pytest.approx(10./4)
    assert solution[y] == pytest.approx(10./4)

    solution = polytope.maximize(5*x + y)
    assert solution[x] == pytest.approx(10./3)
    assert solution[y] == pytest.approx(0)


def test_simple_integer():
    x = Variable('x', lo=0, type=int)
    y = Variable('y', lo=0, type=int)

    polytope = (x + 3*y <= 10) & \
               (3*x + y <= 10)

    solution = polytope.maximize(x+y)
    assert solution[x] == pytest.approx(10//4)
    assert solution[y] == pytest.approx(10//4)

    solution = polytope.maximize(5*x + y)
    assert solution[x] == pytest.approx(3)
    assert solution[y] == pytest.approx(1)


def test_quotient():
    x = Variable('x', lo=0)
    y = Variable('y', lo=0)

    polytope = (1 / y >= 2) & \
               (y / (1 - x) <= 1)

    solution = polytope.maximize(x + 2*y)
    assert solution[x] == pytest.approx(0.5)
    assert solution[y] == pytest.approx(0.5)


def test_or():
    x = Variable('x', lo=0, hi=100)
    y = Variable('y', lo=0, hi=100)

    polytope = ((x <= 10) & (y <= 1)) | \
               ((x <= 3) & (y <= 9))

    solution = polytope.maximize(x+y)
    assert solution[x] == pytest.approx(3)
    assert solution[y] == pytest.approx(9)

    solution = polytope.maximize(2*x+y)
    assert solution[x] == pytest.approx(10)
    assert solution[y] == pytest.approx(1)


def test_or_equality():
    x = Variable('x', lo=0, hi=100)
    polytope = (x == 3) | (x == 5)

    solution = polytope.maximize(x)
    assert solution[x] == pytest.approx(5)

    solution = polytope.minimize(x)
    assert solution[x] == pytest.approx(3)


def test_abs():
    x = Variable('x', lo=0)
    y = Variable('y', lo=0)

    polytope = (x + 3*y <= 10) & \
               (3*x + y <= 10)

    solution = polytope.minimize(abs(x-7))
    assert solution[x] == pytest.approx(10./3)
    assert solution[y] == pytest.approx(0.0)

    solution = polytope.minimize(abs(y-7))
    assert solution[x] == pytest.approx(0.0)
    assert solution[y] == pytest.approx(10./3)

    solution = polytope.minimize(abs(x-7) + abs(y-7))
    assert solution[x] == pytest.approx(10./4)
    assert solution[y] == pytest.approx(10./4)


def test_categorical():
    x = CategoricalVariable('x', ['A', 'B', 'C'])
    y = CategoricalVariable('y', ['A', 'B', 'C'])

    polytope = (x == 'C') & (x == y)
    solution = polytope.minimize(1)

    # TODO: this is really clunky syntax, let's fix it later
    assert solution[x._vars['C']] == 1
    assert solution[y._vars['C']] == 1
