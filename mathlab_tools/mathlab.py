# =============================================================================
# Imports
# =============================================================================

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar

# =============================================================================
# Symbols
# =============================================================================

x, y, z, t, s, u = sp.symbols("x y z t s u", real=True)

# =============================================================================
# Matrix operations
# =============================================================================

def mat(A, dtype=float):
    return np.array(A, dtype=dtype)

M = sp.Matrix
smat = sp.Matrix

det = np.linalg.det
inv = np.linalg.inv
pinv = np.linalg.pinv

eig = np.linalg.eig
eigvals = np.linalg.eigvals

solve = np.linalg.solve

rank = np.linalg.matrix_rank
norm = np.linalg.norm

dot = np.dot
cross = np.cross

eye = np.eye
zeros = np.zeros
ones = np.ones

linspace = np.linspace
logspace = np.logspace

diag = np.diag

def trans(A):
    return np.asarray(A).T

def ctrans(A):
    return np.asarray(A).conj().T

def tr(A):
    return np.trace(A)

# =============================================================================
# Symbolic math
# =============================================================================

def syms(names, **assumptions):
    return sp.symbols(names, **assumptions)

def diff(f, var=x, n=1):
    return sp.diff(f, var, n)

def integ(f, var=x, a=None, b=None):

    if a is None or b is None:
        return sp.integrate(f, var)

    return sp.integrate(f, (var, a, b))

simplify = sp.simplify
expand = sp.expand
factor = sp.factor

pretty = sp.pprint

# =============================================================================
# Vector calculus
# MATLAB-style
# =============================================================================

def vec(Fx, Fy, Fz):

    return sp.Matrix([
        Fx,
        Fy,
        Fz
    ])

def grad(f, vars=(x, y, z)):

    return sp.Matrix([
        sp.diff(f, v)
        for v in vars
    ])

def div(F, vars=(x, y, z)):

    return (
        sp.diff(F[0], vars[0]) +
        sp.diff(F[1], vars[1]) +
        sp.diff(F[2], vars[2])
    )

def curl(F, vars=(x, y, z)):

    X, Y, Z = vars

    Fx, Fy, Fz = F

    return sp.Matrix([
        sp.diff(Fz, Y) - sp.diff(Fy, Z),
        sp.diff(Fx, Z) - sp.diff(Fz, X),
        sp.diff(Fy, X) - sp.diff(Fx, Y)
    ])

def laplacian(f, vars=(x, y, z)):

    return sum(
        sp.diff(f, v, 2)
        for v in vars
    )

# =============================================================================
# Polynomials
# =============================================================================

roots = np.roots
polyval = np.polyval
polyfit = np.polyfit

# =============================================================================
# Numerical utilities
# =============================================================================

def interp1(xdata, ydata, xq, kind="linear"):

    return interp1d(
        xdata,
        ydata,
        kind=kind,
        fill_value="extrapolate"
    )(xq)

def fzero(f, bracket):

    return root_scalar(
        f,
        bracket=bracket
    ).root

def ode45(f, tspan, y0, **kwargs):

    return solve_ivp(
        f,
        tspan,
        y0,
        method="RK45",
        **kwargs
    )

# =============================================================================
# Plotting
# =============================================================================

def fplot(expr, var=x, a=-10, b=10, n=500):

    xs = np.linspace(a, b, n)

    f = sp.lambdify(var, expr, "numpy")

    plt.figure()
    plt.plot(xs, f(xs))
    plt.grid(True)
    plt.show()

# =============================================================================
# Cleanup
# =============================================================================

def chop(A, tol=1e-12):

    A = np.array(A, dtype=float)

    A[np.abs(A) < tol] = 0

    return A

# =============================================================================
# Help
# =============================================================================

def cheatsheet():

    print("""
Matrices:
  det(A)
  inv(A)
  eig(A)
  solve(A,b)

Symbolic:
  diff(f,x)
  integ(f,x)
  simplify(f)

Vector Calculus:
  F = vec(x**2*z,
          y**2*x,
          y+2*z)

  grad(f)
  div(F)
  curl(F)
  laplacian(f)

Polynomials:
  roots()
  polyfit()
  polyval()

Numerical:
  interp1()
  fzero()
  ode45()

Plotting:
  fplot()
""")

def examples():

    print("\\nMatrix Example")
    A = mat([[1,2],[3,4]])
    print(det(A))

    print("\\nCurl Example")
    F = vec(
        x**2*z,
        y**2*x,
        y+2*z
    )

    pretty(curl(F))

    print("\\nGradient Example")
    f = x**2 + y**2 + z**2

    pretty(grad(f))

    print("\\nType cheatsheet()")