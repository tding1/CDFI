import torch
from torch.optim.optimizer import Optimizer, required


class OBProxSG(Optimizer):
    def __init__(
        self,
        params,
        lr=required,
        lambda_=required,
        epochSize=required,
        Np=required,
        No="inf",
        eps=0.0,
    ):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if lambda_ is not required and lambda_ < 0.0:
            raise ValueError("Invalid lambda: {}".format(lambda_))
        if Np is not required and Np < 0.0:
            raise ValueError("Invalid Np: {}".format(Np))
        if epochSize is not required and epochSize < 0.0:
            raise ValueError("Invalid epochSize: {}".format(epochSize))

        self.Np = Np
        self.No = No
        self.epochSize = epochSize
        self.step_count = 0
        self.iter = 0

        defaults = dict(lr=lr, lambda_=lambda_, eps=eps)
        super(OBProxSG, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        No = float("inf") if self.No == "inf" else self.No
        if self.step_count % (self.Np + No) < self.Np:
            doNp = True
            if self.iter == 0:
                print("Prox-SG Step")
        else:
            doNp = False
            if self.iter == 0:
                print("Orthant Step")

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad_f = p.grad.data

                if doNp:
                    s = self.calculate_d(p.data, grad_f, group["lambda_"], group["lr"])
                    p.data.add_(1, s)
                else:
                    state = self.state[p]
                    if "zeta" not in state.keys():
                        state["zeta"] = torch.zeros_like(p.data)
                    state["zeta"].zero_()
                    state["zeta"][p > 0] = 1
                    state["zeta"][p < 0] = -1

                    hat_x = self.gradient_descent(
                        p.data, grad_f, state["zeta"], group["lambda_"], group["lr"]
                    )
                    proj_x = self.project(hat_x, state["zeta"], group["eps"])
                    p.data.copy_(proj_x.data)

        self.iter += 1
        if self.iter >= self.epochSize:
            self.step_count += 1
            self.iter = 0

    def calculate_d(self, x, grad_f, lambda_, lr):
        trial_x = torch.zeros_like(x)
        pos_shrink = x - lr * grad_f - lr * lambda_
        neg_shrink = x - lr * grad_f + lr * lambda_
        pos_shrink_idx = pos_shrink > 0
        neg_shrink_idx = neg_shrink < 0
        trial_x[pos_shrink_idx] = pos_shrink[pos_shrink_idx]
        trial_x[neg_shrink_idx] = neg_shrink[neg_shrink_idx]
        d = trial_x - x
        return d

    def gradient_descent(self, x, grad_f, zeta, lambda_, lr):
        grad = torch.zeros_like(grad_f)
        grad[zeta > 0] = grad_f[zeta > 0] + lambda_
        grad[zeta < 0] = grad_f[zeta < 0] - lambda_
        hat_x = x - lr * grad
        return hat_x

    def project(self, trial_x, zeta, eps):
        proj_x = torch.zeros_like(trial_x)
        keep_indexes = (trial_x * zeta) > eps
        proj_x[keep_indexes] = trial_x[keep_indexes]
        return proj_x
