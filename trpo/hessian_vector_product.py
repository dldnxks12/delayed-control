"""

Hessian 없이 Hessian-Vector Product 구하기 : https://velog.io/@veglog/Hessian-vector-product
Hessian을 이용해야할 때가 있다. 문제는 NN과 같이 거대한 모델의 경우 Hessian을 계산해서 메모리에 올려놓는게 쉽지 않다는 것.

다행히 TRPO에서는 Hessian 말고, Hessian-Vector product(Fisher-Vector product)의 값 그 자체를 이용하기 때문에
이 값을 근사없이 간단하게 구할 수 있는 방법이다.

Hv = ∇(∇L(w)v)


# example
L(w) = 1/2 wAw

w : parameter
A : Hessian

-> Hv = ∇(∇L(w)*v) check
"""


import torch
import torch.nn as nn

torch.manual_seed(0)             # fix seed
H = torch.randn(5, 5)
H = torch.matmul(H, H.T)         # 5x5 대칭행렬
w = nn.Parameter(torch.randn(5)) # parameter
v = torch.randn(5)               # Hessian에 곱할 벡터

L = 0.5 * torch.matmul(torch.matmul(w,H), w)

L_grad = torch.autograd.grad(L, w, create_graph=True)[0]
L_v = torch.matmul(L_grad,v)
L_grad_grad = torch.autograd.grad(L_v, w)[0]

print(L_grad_grad)
result = torch.matmul(H, v)
print(result)

"""
tensor([ -8.4135, -12.9879,   8.0082,  21.6127, -23.2350])
tensor([ -8.4135, -12.9879,   8.0082,  21.6127, -23.2350])
"""








