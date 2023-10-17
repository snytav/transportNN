import torch
from torch.autograd import grad
from torch.autograd.functional import jacobian
from torch.autograd.functional import hessian



def loss_function(x_space, t_space, func, f,c):

    nx = x_space.shape[0]
    nt = t_space.shape[0]

    mloss   = torch.zeros(nt,nx)
    m_df_dx = torch.zeros(nt,nx)
    m_df_dt = torch.zeros(nt,nx)
    loss_sum = 0.0
    for i,xi in enumerate(x_space):
        for k,ti in enumerate(t_space):
            input_point = torch.Tensor([xi,  ti])
            input_point.requires_grad_()

            #net_out = pde.forward(input_point)
            # net_out_w = grad(outputs=net_out, inputs=pde.fc1.weight, grad_outputs=torch.ones_like(net_out),
            #                  retain_graph=True, create_graph=True)

            net_out_jacobian = jacobian(func, input_point, create_graph=True)
            # jac1  = get_jacobian(pde.forward,input_point,2)
            # net_out_hessian = hessian(pde.forward, input_point, create_graph=True)
            psy_t = func(input_point)

            inputs = input_point
            psy_t_jacobian = jacobian(func, inputs, create_graph=True)
            # psy_t_hessian = hessian(psy_trial, inputs, create_graph=True)
            # psy_t_hessian = psy_t_hessian[0][0]
            # acobian(jacobian(psy_trial))(input_point, net_out

            trial_dx = psy_t_jacobian[0]
            trial_dt = psy_t_jacobian[1]

            rhs_func = f(input_point)
            func_t = torch.Tensor([rhs_func])
            func_t.requires_grad_()

            err_sqr = ((trial_dt + c * trial_dx) - func_t) ** 2
            # D_err_sqr_D_W0 = 2*((gradient_of_trial_d2x + gradient_of_trial_d2y) - func)*(
            #                     (D_gradient_of_trial_d2x_D_W0 + D_gradient_of_trial_d2y_D_W0) -D_func_D_W0
            #                     )
            mloss[k][i] = err_sqr
            m_df_dx[k][i] = trial_dx
            m_df_dt[k][i] = trial_dt

            loss_sum += err_sqr
            qq = 0

        return loss_sum,mloss,m_df_dx,m_df_dt
