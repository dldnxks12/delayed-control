import torch

# retain_graph = False -> the graph used to compute the grad will be freed
# create_graph = True  -> graph of the derivative will be constructed, allowing to compute higher order derivative products

def f(x):
    return x*x*torch.arange(4, dtype = torch.float)

def jacobian(y, x, create_graph = False): # y : f(x)
    jacob = []
    flat_y = y.reshape(-1)            # [0, 1, 2, 3]
    grad_y = torch.zeros_like(flat_y) # [0, 0, 0, 0]

    for i in range(len(flat_y)):
        grad_y[i] = 1

        grad_x, = torch.autograd.grad(flat_y, x, grad_y, retain_graph=True, create_graph=create_graph)
        jacob.append(grad_x.reshape(x.shape))
        grad_y[i] = 0

    return torch.stack(jacob).reshape(y.shape + x.shape) # [4,4]

def hessian(y, x):
    return jacobian(jacobian(y, x, create_graph=True), x)

x = torch.ones((4, 4), requires_grad=True)
J = jacobian(f(x), x)
H = hessian(f(x), x)





