from torch.autograd import Function


class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        max_norm = 0.5
        norm_type = 2
        param_norm = grads.data.norm(norm_type).item()
        clip_coeff = max_norm / (param_norm + 1e-6)

        if clip_coeff < 1:
            grads = grads * clip_coeff

        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None


def grad_reverse(x, lambd):
    return GradientReversalFunction.apply(x, lambd)
