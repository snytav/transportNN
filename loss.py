import torch
from torch.autograd import grad
from torch.autograd.functional import jacobian
from torch.autograd.functional import hessian



def loss_function(x_space, t_space, pde, psy_trial, f,c):

    loss_sum = 0.0
    for xi in x_space:
        for ti in t_space:
            input_point = torch.Tensor([xi,  ti])
            input_point.requires_grad_()

            net_out = pde.forward(input_point)
            # net_out_w = grad(outputs=net_out, inputs=pde.fc1.weight, grad_outputs=torch.ones_like(net_out),
            #                  retain_graph=True, create_graph=True)

            net_out_jacobian = jacobian(pde.forward, input_point, create_graph=True)
            # jac1  = get_jacobian(pde.forward,input_point,2)
            # net_out_hessian = hessian(pde.forward, input_point, create_graph=True)
            psy_t = psy_trial(input_point, net_out)

            inputs = (input_point, net_out)
            psy_t_jacobian = jacobian(psy_trial, inputs, create_graph=True)[0]
            # psy_t_hessian = hessian(psy_trial, inputs, create_graph=True)
            # psy_t_hessian = psy_t_hessian[0][0]
            # acobian(jacobian(psy_trial))(input_point, net_out

            trial_dx = psy_t_jacobian[0][0][0]
            trial_dt = psy_t_jacobian[0][0][1]

            func = f(input_point)
            func_t = torch.Tensor([func])
            func_t.requires_grad_()

            err_sqr = ((trial_dt + c * trial_dx) - func_t) ** 2
            # D_err_sqr_D_W0 = 2*((gradient_of_trial_d2x + gradient_of_trial_d2y) - func)*(
            #                     (D_gradient_of_trial_d2x_D_W0 + D_gradient_of_trial_d2y_D_W0) -D_func_D_W0
            #                     )

            loss_sum += err_sqr
            qq = 0

        return loss_sum
