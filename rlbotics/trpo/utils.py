import torch

def kl_div(p, q):
    p = p.detach()
    return (p * (p.log() - q.log())).sum(-1).mean()

def flat_grad(y, x, retain_graph=False, create_graph=False):
    if create_graph:
        retain_graph = True

    g = torch.autograd.grad(y, x, retain_graph=retain_graph, create_graph=create_graph)
    g = torch.cat([t.view(-1) for t in g])
    return g

def conjugate_gradient(f_Ax, b, cg_iters=10, callback=None, residual_tol=1e-10):
    p = b.clone()
    r = b.clone()
    x = torch.zeros_like(b)

    rdotr = torch.dot(r, r)

    for i in range(cg_iters):
        if callback is not None:
            callback(x)
        z = f_Ax(p)
        v = rdotr / torch.dot(p, z)
        x += v*p
        r -= v*z
        newrdotr =torch.dot(r, r)
        mu = newrdotr/rdotr
        p = r + mu*p

        rdotr = newrdotr
        if rdotr < residual_tol:
            break

    if callback is not None:
        callback(x)
    return x
