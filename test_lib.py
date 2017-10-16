import pytest
from lib import *

def test_simple():
    x = Variable('x', lo=0)
    y = Variable('y', lo=0)

    polytope = (x + 3*y <= 5) & \
               (3*x + y <= 5)
    problem = Problem(polytope, x+y)
    solution = problem.solve()

    assert solution[x] == pytest.approx(1.25)
    assert solution[y] == pytest.approx(1.25)
