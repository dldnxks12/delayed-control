"""
https://joonleesky.github.io/posts/Conjugate_Gradient/

Ax = b 의 문제를 해결하는 방법

1. A의 역행렬을 구해 양변에 곱하는 방법
    - general method (Neural network의 hessian의 역행렬을 구하기 매우 어려우므로 이 방법은 X)

2. Gaussian elimination
    - n개의 연립방정식에 대해 O(n^3) 안에 해결 가능

3. optimize 관점으로 해결하는 conjugate gradient method
    - A의 역행렬을 구해 x를 직접 구하지 않고, Ax = b의 해를 근사하는 방법으로 찾는다. (최소점을 찾는 문제로 환원해서 gradient descent!)
    - n번 안에 정확한 해를 찾을 수 있음
    - A-conjugate vector 찾기!


* conjugate gradient method를 사용하기 위한 조건

    1. A => R(nxn) square matrix
    2. A => positive definite matrix

        Ex) Identity matrix, Invertible matrix, Hessian


"""
import sys
import torch
import numpy as np

# Define Positive definite matrix
n = 1000
A = np.random.normal(size = [n,n])
A = np.dot(A.T, A)    # 100 x 100
b = np.random.rand(n) # 100,

# Gradient descent
x = np.zeros(n)
for i in range(n):
    r     = b - np.dot(A, x) # residual
    alpha = np.dot(r, r) / np.dot(r, np.dot(A, r))
    x     = x + alpha*r


# Conjugate gradient descent
x = np.zeros(n)
r = b - np.dot(A, x)  # residual
v = np.copy(r)        # search direction
damping = 0.1
for i in range(n):
    Av    = np.dot(A, v) + damping * v
    alpha = np.dot(v.T, r) / np.dot(v.T, Av)
    x     = x + alpha*v  # x update
    r     = r - alpha*Av # residual update

    v     = r - (np.dot(np.dot(r.T, A), v) / np.dot(Av, v))*v # gram-schmidt
    print(r.sum())




