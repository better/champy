from ortools.linear_solver import pywraplp

class Expression:
    def __init__(self, expr, constant=0):
        self._expr = expr
        self._constant = constant

    def __mul__(self, rhs):
        assert type(rhs) in [float, int]
        return Expression(tuple((k*rhs, v) for k, v in self._expr))

    def __rmul__(self, lhs):
        return self * lhs

    def __add__(self, rhs):
        if type(rhs) in [float, int]:
            return Expression(self._expr, self._constant + rhs)
        else:
            assert isinstance(rhs, Expression)
            coeffs = {}
            vs = {}
            for c, v in self._expr:
                coeffs[v._id] = c
                vs[v._id] = v
            for c, v in rhs._expr:
                coeffs[v._id] = coeffs.get(v._id, 0) + c
                vs[v._id] = v
            return Expression(tuple((coeffs[v._id], v) for v in vs.values()), self._constant + rhs._constant)

    def __radd__(self, lhs):
        return self * lhs

    def __neg__(self):
        return self * -1

    def __sub__(self, rhs):
        return self + (-rhs)

    def __rsub__(self, lhs):
        return lhs + (-self)

    def __abs__(self):
        return Problem.abs(self)
    
    #def __lt__(self, rhs):
    #    return Polytope(((-self - rhs, '>'),))

    def __le__(self, rhs):
        return Polytope(((-self - rhs, '>='),))

    #def __gt__(self, rhs):
    #    return Polytope(((self - rhs, '>'),))

    def __ge__(self, rhs):
        return Polytope(((self - rhs, '>='),))

    def __eq__(self, rhs):
        return Polytope(((self - rhs, '=='),))

    def __str__(self):
        return 'Expression(%s+%.2f)' % (''.join('%+.2f%s' % (k, v) for k, v in self._expr), self._constant)


class Variable(Expression):
    _id = 0
    def __init__(self, name=None, lo=None, hi=None, cat=None):
        self._name = name
        self._expr = ((1, self),)
        self._constant = 0
        self._lo = lo
        self._hi = hi
        self._cat = cat
        Variable._id += 1
        self._id = Variable._id

    def __str__(self):
        return self._name


class Polytope:
    def __init__(self, constraints):
        self._constraints = tuple(constraints)

    @staticmethod
    def all(polytopes):
        return Polytope(sum((c._constraints for c in polytopes), tuple()))

    @staticmethod
    def any(polytopes, big_M=999999):
        # This one is the most interesting
        magic = [Variable('magic', 0, 1, 'binary') for p in polytopes]
        new_constraints = []
        for polytope, m in zip(polytopes, magic):
            for expr, op in polytope._constraints:
                new_constraints.append((expr + m * big_M, op))
        return Polytope(new_constraints) & \
            (sum(magic) <= len(polytopes)-1)

    def __and__(self, rhs):
        return Polytope.all((self, rhs))

    def __or__(self, rhs):
        return Polytope.any((self, rhs))
    
    def __str__(self):
        return 'Polytope(%s)' % ', '.join('%s %s 0' % (str(expr), op) for expr, op in self._constraints)


class Problem:
    def __init__(self, polytope, objective):
        self._polytope = polytope
        self._objective = objective

    @staticmethod
    def abs(m):
        z_neg = Variable('z_neg', lo=0, hi=None)
        z_pos = Variable('z_pos', lo=0, hi=None)
        return Problem(m + z_neg + z_pos == 0, z_neg + z_pos)

    def solve(self):
        solver = pywraplp.Solver('', pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)
        vs = {}
        objective = solver.Objective()
        def get_var(v):
            if v._id not in vs:
                vs[v._id] = solver.NumVar(v._lo if v._lo is not None else -solver.infinity(),
                                          v._hi if v._hi is not None else -solver.infinity(),
                                          '%s_%s' % (v._name, v._id))
            return vs[v._id]
            
        for k, v in self._objective._expr:
            objective.SetCoefficient(get_var(v), k)
        for c, op in self._polytope._constraints:
            if op == '==':
                constraint = solver.Constraint(0, 0)
            elif op == '>=':
                constraint = solver.Constraint(0, solver.infinity())
            for k, v in c._expr:
                constraint.SetCoefficient(get_var(v), k)
        print(solver.Solve())
                

    def __str__(self):
        return 'Problem(min %s s.t. %s)' % (self._objective, self._polytope)

x = Variable('x')
y = Variable('y')
z = Variable('z')
c1 = 3 * (x - y*2) >= 7 * x
c2 = z <= y
print(c1 & c2)
print(c1 | c2)
abs(x - y).solve()
