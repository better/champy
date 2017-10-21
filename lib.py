from ortools.linear_solver import pywraplp

class Expression:
    def __init__(self, expr, constant=0, polytope=None):
        self._expr = expr
        self._constant = constant
        if polytope is None:
            polytope = Polytope()
        self._polytope = polytope # only used for certain ones tied to the definition (see abs)

    def __mul__(self, rhs):
        assert type(rhs) in [float, int]
        return Expression(
            tuple((k*rhs, v) for k, v in self._expr),
            self._constant*rhs,
            self._polytope
        )

    def __rmul__(self, lhs):
        return self * lhs

    def __truediv__(self, rhs):
        if type(rhs) in [float, int]:
            return self * (1./rhs)
        else:
            assert isinstance(rhs, Expression)
            return QuotientExpression(self, rhs)

    def __rtruediv__(self, lhs):
        if type(lhs) in [float, int]:
            lhs = Expression(tuple(), lhs)
        assert isinstance(lhs, Expression)
        return QuotientExpression(lhs, self)

    def __add__(self, rhs):
        if type(rhs) in [float, int]:
            return Expression(
                self._expr,
                self._constant + rhs,
                self._polytope
            )
        else:
            assert isinstance(rhs, Expression)
            vs = {}
            for c, v in self._expr:
                vs[v] = vs.get(v, 0) + c
            for c, v in rhs._expr:
                vs[v] = vs.get(v, 0) + c
            return Expression(
                tuple((c, v) for v, c in vs.items()),
                self._constant + rhs._constant,
                self._polytope & rhs._polytope
            )

    def __radd__(self, lhs):
        return self + lhs

    def __neg__(self):
        return self * -1

    def __sub__(self, rhs):
        return self + (-rhs)

    def __rsub__(self, lhs):
        return lhs + (-self)
    
    #def __lt__(self, rhs):
    #    return Polytope(((-self - rhs, '>'),))

    def __le__(self, rhs):
        return Polytope(((rhs - self, '>='),))

    #def __gt__(self, rhs):
    #    return Polytope(((self - rhs, '>'),))

    def __ge__(self, rhs):
        return Polytope(((self - rhs, '>='),))

    def __eq__(self, rhs):
        return Polytope(((self - rhs, '=='),))

    def __abs__(self):
        z_neg = Scalar('z_neg', lo=0, hi=None)
        z_pos = Scalar('z_pos', lo=0, hi=None)
        return Expression(
            ((1, z_neg), (1, z_pos)),
            0,
            self + z_neg + z_pos == 0)

    def min(self):
        # Compute a lower bound of the expression
        s = self._constant
        for k, v in self._expr:
            if k > 0 and v._lo is None:
                return None
            elif k > 0:
                s += k * v._lo
            elif k < 0 and v._hi is None:
                return None
            elif k < 0:
                s += k * v._hi
        return s

    def __str__(self):
        return 'Expression(%s+%.2f)' % (''.join('%+.2f%s' % (k, v) for k, v in self._expr), self._constant)

    def __repr__(self):
        return str(self)


class QuotientExpression:
    # Not an Expression subclass. Can only be used to compare with constants.
    # Useful for saying eg (x + 5) / (y + 2) <= 3
    def __init__(self, nom, den):
        self._nom = nom
        self._den = den

    def __le__(self, rhs):
        return Polytope(((self._den * rhs - self._nom, '>='),))

    def __ge__(self, rhs):
        return Polytope(((self._nom - self._den * rhs, '>='),))

    def __eq__(self, rhs):
        return Polytope(((self._den * rhs - self._nom, '=='),))


class Variable:
    # Provides a higher level abstractions on top of variables
    pass


class Scalar(Variable, Expression):
    _SELF = object()
    def __init__(self, name='untitled', lo=None, hi=None, type=float, variable=_SELF):
        self._name = name
        if type == bool:
            lo, hi = 0, 1
        self._lo = lo
        self._hi = hi
        self._type = type
        if variable is Scalar._SELF:
            variable = self
        self._variable = variable
        super(Scalar, self).__init__(((1, self),))

    def variable(self):
        return self._variable

    def value(self, scalars):
        return scalars[self]

    def __hash__(self):
        return id(self)

    def __str__(self):
        return 'Variable#%s' % self._name

    def __repr__(self):
        return str(self)


class Categorical(Variable):
    def __init__(self, name='untitled', options=[]):
        self._name = name
        self._options = options
        self._vars = {o: Scalar(name='[%s==%s]' % (name, o), type=bool, variable=self) for o in options}
        polytope = sum(self._vars.values()) == 1
        for v in self._vars.values():
            v._polytope = polytope

    def value(self, scalars):
        for o, scalar in self._vars.items():
            if scalars[scalar] == 1:
                return o

    def __eq__(self, rhs):
        if isinstance(rhs, Categorical):
            assert self._options == rhs._options
            return Polytope.all(self._vars[o] == rhs._vars[o] for o in self._options)
        else:
            assert rhs in self._vars
            return self._vars[rhs] == 1

    def __ne__(self, rhs):
        if isinstance(rhs, Categorical):
            assert self._options == rhs._options
            return Polytope.any((self._vars[o] == 1) & (rhs._vars[o] == 0)
                                for o in self._options)
        else:
            assert rhs in self._vars
            return self != self._options[rhs]

    def __hash__(self):
        return id(self)

    def __str__(self):
        return 'Categorical#%s' % self._name

    def __repr__(self):
        return str(self)

class Polytope:
    def __init__(self, constraints=tuple()):
        self._constraints = tuple(constraints)

    @staticmethod
    def all(polytopes):
        return Polytope(sum((c._constraints for c in polytopes), tuple()))

    @staticmethod
    def any(polytopes, big_M=None):
        # This one is the most interesting
        # The idea is to hand out n-1 "free cards" where a free card magically solves the inequality
        polytopes = tuple(polytopes)
        magic = [Scalar('magic', type=bool, variable=None) for p in polytopes]
        new_constraints = []
        for polytope, m in zip(polytopes, magic):
            for expr, op in polytope._constraints:
                if op == '==':
                    # Rewrite equality to two inequalities
                    exprs = [expr, -expr]
                else:
                    exprs = [expr]
                for expr in exprs:
                    if big_M is None:
                        lower_bound = expr.min()
                        if lower_bound is None:
                            raise Exception('big_M is not provided and %s is unbounded' % expr)
                        else:
                            magic_term = -lower_bound
                    else:
                        magic_term = big_M
                    new_constraints.append((expr + m * magic_term, '>='))

        return Polytope(new_constraints) & (sum(magic) <= len(polytopes)-1)

    def __and__(self, rhs):
        return Polytope.all((self, rhs))

    def __or__(self, rhs):
        return Polytope.any((self, rhs))
    
    def optimize(self, objective, sign):
        if not isinstance(objective, Expression):
            # Not a function of the variables, so let's just replace it with a constant 0
            objective = Expression(tuple())
        ot_solver = pywraplp.Solver('test', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
        variables = set()
        ot_variables = {}
        ot_variable_name_count = {}
        ot_objective = ot_solver.Objective()
        cls = {int: lambda lo, hi, name: ot_solver.IntVar(lo, hi, name),
               float: lambda lo, hi, name: ot_solver.NumVar(lo, hi, name),
               bool: lambda lo, hi, name: ot_solver.BoolVar(name)}
        def get_ot_var(v):
            if v not in ot_variables:
                if v._name in ot_variable_name_count:
                    name = '%s[%d]' % (v._name, ot_variable_name_count[v._name])
                else:
                    name = v._name
                ot_variable_name_count[v._name] = ot_variable_name_count.get(v._name, 0) + 1
                ot_variables[v] = cls[v._type](v._lo if v._lo is not None else -ot_solver.infinity(),
                                               v._hi if v._hi is not None else ot_solver.infinity(),
                                               name)
                variable = v.variable()
                if variable is not None:
                    variables.add(variable)
            return ot_variables[v]

        for k, v in objective._expr:
            ot_objective.SetCoefficient(get_ot_var(v), k * sign)
        ot_objective.SetMaximization()
        added_polytopes = set()
        def add_polytope(p):
            if p in added_polytopes:
                return
            added_polytopes.add(p)
            for c, op in p._constraints:
                if op == '==':
                    ot_constraint = ot_solver.Constraint(-c._constant, -c._constant)
                elif op == '>=':
                    ot_constraint = ot_solver.Constraint(-c._constant, ot_solver.infinity())
                for k, v in c._expr:
                    ot_constraint.SetCoefficient(get_ot_var(v), k)
                if c._polytope:
                    add_polytope(c._polytope)
        add_polytope(self)
        if objective._polytope:
            add_polytope(objective._polytope)
        res = ot_solver.Solve()
        scalar_values = {}
        for v, nv in ot_variables.items():
            scalar_values[v] = nv.solution_value()
        solution = {}
        for v in variables:
            solution[v] = v.value(scalar_values)
        return solution

    def maximize(self, objective):
        return self.optimize(objective, 1)

    def minimize(self, objective):
        return self.optimize(objective, -1)

    def __str__(self):
        return 'Polytope(%s)' % ', '.join('%s %s 0' % (str(expr), op) for expr, op in self._constraints)

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return id(self)
