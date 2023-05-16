"""
https://towardsdatascience.com/complete-step-by-step-gradient-descent-algorithm-from-scratch-acba013e8420
"""

import numpy as np

def ArmijoLineSearch(f, xk, pk, gfk, phi0, alpha0, rho = 0.5, c1 = 1e-4):

    """
    Minimize over alpha. the function f(xk + alpha*pk) <= f(xk) + c1*alpha*∇fk*pk

    f   : function to be minimized

    xk  : current point
    pk  : search direction

    gfk  : gradient of f at point x(k)
    phi0 : value of f at point x(k)

    alpha0 : value of initial alpha
    rho    : value of shrinkage factor
    c1     : value to control stopping criterion

    """

    derphi0 = np.dot(gfk, pk)
    phi_a0  = f(xk + alpha0*pk)

    while not phi_a0 <= phi0 + c1*alpha0*derphi0:
        alpha0 = alpha0 * rho
        new_xk = xk + alpha0+pk
        phi_a0 = f(new_xk)   # value of f at point of new x(k)

    # return final step size alpha0
    # return value of f at new point x(k+1)
    return alpha0, phi_a0


def GradientDescent(f, f_grad, init, alpha=1, tol=1e-5, max_iter=1000):
    """Gradient descent method for unconstraint optimization problem.
    given a starting point x ∈ Rⁿ,
    repeat
        1. Define direction. p := −∇f(x).
        2. Line search. Choose step length α using Armijo Line Search.
        3. Update. x := x + αp.
    until stopping criterion is satisfied.

    Parameters
    --------------------
    f : callable
        Function to be minimized.
    f_grad : callable
        The first derivative of f.
    init : array
        initial value of x.
    alpha : scalar, optional
        the initial value of steplength.
    tol : float, optional
        tolerance for the norm of f_grad.
    max_iter : integer, optional
        maximum number of steps.

    Returns - for visualize
    --------------------
    xs : array
        x in the learning path
    ys : array
        f(x) in the learning path
    """

    # initialize x, f(x), and f'(x)
    xk = init
    fk = f(xk)
    gfk = f_grad(xk)
    gfk_norm = np.linalg.norm(gfk)

    # initialize number of steps, save x and f(x)
    num_iter = 0
    curve_x = [xk]
    curve_y = [fk]
    print('Initial condition: y = {:.4f}, x = {} \n'.format(fk, xk))
    # take steps
    while gfk_norm > tol and num_iter < max_iter:
        # determine direction
        pk = -gfk
        # calculate new x, f(x), and f'(x)
        alpha, fk = ArmijoLineSearch(f, xk, pk, gfk, fk, alpha0=alpha)
        xk = xk + alpha * pk
        gfk = f_grad(xk)
        gfk_norm = np.linalg.norm(gfk)
        # increase number of steps by 1, save new x and f(x)
        num_iter += 1
        curve_x.append(xk)
        curve_y.append(fk)
        print('Iteration: {} \t y = {:.4f}, x = {}, gradient = {:.4f}'.
              format(num_iter, fk, xk, gfk_norm))
    # print results
    if num_iter == max_iter:
        print('\nGradient descent does not converge.')
    else:
        print('\nSolution: \t y = {:.4f}, x = {}'.format(fk, xk))

    return np.array(curve_x), np.array(curve_y)

