"""
# ref : https://sungwookyoo.github.io/study/conjugate_gradient/

* gradient descent : learning rate를 고정시킨 채로 gradient를 이용해서 이동
* steepest descent : line search를 이용해서 learning rate를 갱신

"""

import sys
import torch
import numpy as np

def conjugate(A, b, init, max_step):
    x = init.copy()

    r = b - np.matmul(A, x) # gradient of f(x)
    p = r.copy()            # Initial search direction

    bound = 1e-5
    while max_step:

        # -- line search -- #
        # get alpha
        PAP   = np.matmul(np.matmul(p.T, A), p)
        alpha = np.matmul(r.T, r) / PAP

        # update estiamate x
        x_new = x + alpha*p

        if bound > sum(np.abs(alpha*p)):
            break

        r_old = r.copy()
        r     = b - np.matmul(A, x_new)

        beta = np.matmul(r.T, r) / np.matmul(r_old.T, r_old)
        p    = r + beta*p

        x = x_new
        max_step -= 1

        return x








