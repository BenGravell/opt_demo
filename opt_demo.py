import autograd.numpy as np
import autograd.numpy.linalg as la
import autograd.numpy.random as npr
import matplotlib.pyplot as plt
from time import time
import os

from lqr_utility import vec, mat, dlyap, dare, gain, check_are, calc_cost_manual, calc_grad_manual, calc_hess_manual
from lqr_problems import gen_lqr_problem, make_lqr_objective
from lqr_optimizers import value_iteration, policy_iteration, riccati_direct
from optimizers import Objective, \
    GradientOptSetting, GradientOptimizer, \
    LineSearchOptSetting, LineSearchOptimizer, \
    QuasiNewtonOptSetting, QuasiNewtonOptimizer, \
    TrustRegionOptSetting, TrustRegionOptimizer
from scipy.optimize import minimize as sp_minimize
from colors import PrintColors

MIN_GRAD_NORM = 1e-4


def sanity_check(K, A, B, Q, X0, discount=None, f=None, g=None, h=None, tol=1e-6, verbose=True):
    # Sanity check - compare cost, gradient, hessian-quadratic-form with hand-calculated expressions
    n, m = B.shape
    vK = vec(K)

    if verbose:
        print('SANITY CHECK')

    # Cost
    if f is not None:
        C0 = calc_cost_manual(K, A, B, Q, X0, discount)
        C0_true = f(vK)

        if verbose:
            print('cost')
            print(C0_true)
            print(C0)

        if np.abs(C0_true - C0) > tol:
            raise ValueError('Sanity check failed! Cost does not match true value.')

    # Gradient
    if g is not None:
        G0 = vec(calc_grad_manual(K, A, B, Q, X0, discount))
        G0_true = g(vK)

        if verbose:
            print('gradient')
            print(G0_true)
            print(G0)

        if la.norm(G0 - G0_true) > tol:
            raise ValueError('Sanity check failed! Gradient does not match true value.')

    # Hessian
    # (Technically we need to check if Hessian-quadratic-form matches at all possible E,
    # but here we just use a single point E)
    if h is not None:
        E = npr.randn(m, n)
        H0_EE = calc_hess_manual(K, E, A, B, Q, X0, discount)
        H0_EE_true = np.dot(vec(E), np.dot(h(vK), vec(E)))

        if verbose:
            print('hessian quadform')
            print(H0_EE)
            print(H0_EE_true)

        if np.abs(H0_EE - H0_EE_true) > tol:
            raise ValueError('Sanity check failed! Hessian quadform does not match true value,')

    if verbose:
        print('')
    return


def plot_single(result, x_opt, f_opt, method):
    # Unpack result
    t_hist = result['t_hist']
    x_hist = result['x_hist']
    f_hist = result['f_hist']
    g_hist = result['g_hist']
    h_hist = result['h_hist']

    # Plot
    fig, ax = plt.subplots(nrows=4, sharex=True, figsize=(6, 10))

    ax[0].semilogy(t_hist, la.norm(x_hist-x_opt, axis=1))
    ax[0].set_ylabel('Iterate error norm')

    ax[1].semilogy(t_hist, f_hist-f_opt)
    ax[1].set_ylabel('Objective error')

    ax[2].semilogy(t_hist, la.norm(g_hist, axis=1))
    ax[3].axhline(MIN_GRAD_NORM, linestyle='--', color='k', alpha=0.5)
    ax[2].set_ylabel('Gradient norm')

    ax[3].plot(t_hist, np.array([la.eigh(h)[0] for h in h_hist]))
    ax[3].axhline(0, linestyle='--', color='k', alpha=0.5)
    ax[3].set_yscale('symlog')
    ax[3].set_ylabel('Hessian eigenvalues')

    ax[-1].set_xlabel('Iteration')
    ax[0].set_title(method)
    return fig, ax


def plot_multi(results_dict, x_opt, f_opt, category):
    fig, ax = plt.subplots(nrows=3, sharex=True, figsize=(8, 10))
    for method, result_dict in results_dict.items():
        result = result_dict['result']
        # Unpack result
        t_hist = result['t_hist']
        x_hist = result['x_hist']
        f_hist = result['f_hist']
        g_hist = result['g_hist']
        h_hist = result['h_hist']
        elapsed_time = result['elapsed_time']
        label = method + ' (%.3f sec)' % elapsed_time

        ax[0].semilogy(t_hist, la.norm(x_hist-x_opt, axis=1), label=label)
        ax[1].semilogy(t_hist, f_hist-f_opt, label=label)
        ax[2].semilogy(t_hist, la.norm(g_hist, axis=1), label=label)

    ax[2].axhline(MIN_GRAD_NORM, color='k', linestyle='--', alpha=0.5)

    # xlim = (-1.0, ax[0].get_xlim()[1] * 1.7)
    xlim = (-1.0, 71)
    for i in range(3):
        ax[i].legend(ncol=1, loc='upper right')
        ax[i].set_xlim(xlim)
    ax[0].set_title(category+' methods')
    ax[-1].set_xlabel('Iteration')
    ax[0].set_ylabel('Iterate error norm')
    ax[1].set_ylabel('Objective error')
    ax[2].set_ylabel('Gradient norm')
    fig.tight_layout()
    return fig, ax


if __name__ == "__main__":
    # Reset global options
    np.set_printoptions(precision=3)
    seed = 1

    # Make LQR problem data
    A, B, Q, X0 = gen_lqr_problem(n=3, m=2, rho=0.9, round_places=1, seed=seed)
    n, m = B.shape
    discount = None

    # Make LQR objective
    f, g, h = make_lqr_objective(A, B, Q, X0, discount)
    obj = Objective(f, g, h, name='LQR')

    # Initial policy
    K0 = np.zeros([m, n])
    vK0 = vec(K0)

    sanity_check(K0, A, B, Q, X0, discount, f, g, h)

    # Get the baseline solution by solving the Riccati equation directly
    P_are, K_are = riccati_direct(A, B, Q, discount)
    f_are = f(vec(K_are))

    # # SciPy optimization
    # method_list = ['Nelder-Mead',
    #                'Powell',
    #                'CG',
    #                'Newton-CG',
    #                'BFGS',
    #                'dogleg',
    #                'trust-ncg',
    #                'trust-exact',
    #                'trust-krylov']
    # sp_results_dict = {}
    # for method in method_list:
    #     if method in ['Nelder-Mead', 'Powell']:
    #         sp_jac = None
    #     else:
    #         sp_jac = g
    #     if method in ['Nelder-Mead', 'Powell', 'CG', 'BFGS']:
    #         sp_hess = None
    #     else:
    #         sp_hess = h
    #     if method in ['dogleg']:
    #         def h_pos(x, h):
    #             from lqr_utility import posdefify
    #             H = h(x)
    #             return posdefify(H, eps=1e-6)
    #         from functools import partial
    #         sp_hess = partial(h_pos, h=h)
    #
    #     if method == 'Nelder-Mead':
    #         sp_result = sp_minimize(f, x0=vK0, jac=sp_jac, hess=sp_hess, method=method,
    #                                 options={'maxiter': 10000, 'maxfev': 10000,
    #                                          'xatol': 0, 'fatol': 1e-6})
    #     else:
    #         sp_result = sp_minimize(f, x0=vK0, jac=sp_jac, hess=sp_hess, method=method)
    #
    #     sp_results_dict[method] = sp_result
    #
    # for method, sp_result in sp_results_dict.items():
    #     print(method)
    #     if sp_result.message == 'Optimization terminated successfully.':
    #         print(f"{PrintColors.OKGREEN}Optimization converged successfully!{PrintColors.ENDC}")
    #     print(sp_result.x)
    #     print(sp_result['fun'])
    #     print('')

    # From-scratch optimization
    # Optimization settings
    constant_stepsize_gradient_methods = 0.1
    max_iters_gradient_methods = 500
    settings_dict_gradient = {
                     'gradient':
                         {'setting': GradientOptSetting(x0=vK0,
                                                        a0=constant_stepsize_gradient_methods,
                                                        step_method='gradient',
                                                        max_iters=max_iters_gradient_methods,
                                                        verbose_stride=10,
                                                        min_grad_norm=MIN_GRAD_NORM),
                          'optimizer': GradientOptimizer},
                     'momentum':
                         {'setting': GradientOptSetting(x0=vK0,
                                                        a0=constant_stepsize_gradient_methods,
                                                        step_method='momentum',
                                                        mass=0.5,
                                                        v0=np.zeros_like(vK0),
                                                        max_iters=max_iters_gradient_methods,
                                                        verbose_stride=10,
                                                        min_grad_norm=MIN_GRAD_NORM),
                          'optimizer': GradientOptimizer},
                     'nesterov':
                         {'setting': GradientOptSetting(x0=vK0,
                                                        a0=constant_stepsize_gradient_methods,
                                                        step_method='nesterov',
                                                        mass=0.5,
                                                        v0=np.zeros_like(vK0),
                                                        max_iters=max_iters_gradient_methods,
                                                        verbose_stride=10,
                                                        min_grad_norm=MIN_GRAD_NORM),
                          'optimizer': GradientOptimizer},
                     'relativistic':
                         {'setting': GradientOptSetting(x0=vK0,
                                                        a0=constant_stepsize_gradient_methods,
                                                        step_method='relativistic',
                                                        mass=0.5,
                                                        delta=20.0,
                                                        v0=np.zeros_like(vK0),
                                                        max_iters=max_iters_gradient_methods,
                                                        verbose_stride=10,
                                                        min_grad_norm=MIN_GRAD_NORM),
                          'optimizer': GradientOptimizer},
                     'rmsprop':
                         {'setting': GradientOptSetting(x0=vK0,
                                                        a0=constant_stepsize_gradient_methods,
                                                        step_method='rmsprop',
                                                        avg_sq_grad0=np.ones_like(vK0),
                                                        gamma=0.9,
                                                        eps=1e-8,
                                                        max_iters=max_iters_gradient_methods,
                                                        verbose_stride=10,
                                                        min_grad_norm=MIN_GRAD_NORM),
                          'optimizer': GradientOptimizer},
                     'adam':
                         {'setting': GradientOptSetting(x0=vK0,
                                                        a0=constant_stepsize_gradient_methods,
                                                        step_method='adam',
                                                        eps=1e-8,
                                                        b1=0.5,
                                                        b2=0.99,
                                                        mean0=np.zeros_like(vK0),
                                                        variance0=np.zeros_like(vK0),
                                                        max_iters=max_iters_gradient_methods,
                                                        verbose_stride=10,
                                                        min_grad_norm=MIN_GRAD_NORM),
                          'optimizer': GradientOptimizer},
                     }

    settings_dict_linesearch = {
                     'gradient':
                         {'setting': LineSearchOptSetting(x0=vK0,
                                                          a0=1.0,
                                                          step_method='gradient',
                                                          max_iters=1000,
                                                          verbose_stride=100,
                                                          min_grad_norm=MIN_GRAD_NORM),
                          'optimizer': LineSearchOptimizer},
                     'newton':
                         {'setting': LineSearchOptSetting(x0=vK0,
                                                          step_method='newton',
                                                          min_grad_norm=MIN_GRAD_NORM),
                          'optimizer': LineSearchOptimizer},
                     'bfgs':
                         {'setting': QuasiNewtonOptSetting(x0=vK0,
                                                           step_method='bfgs',
                                                           min_grad_norm=MIN_GRAD_NORM),
                          'optimizer': QuasiNewtonOptimizer},
                     'dfp':
                         {'setting': QuasiNewtonOptSetting(x0=vK0,
                                                           step_method='dfp',
                                                           min_grad_norm=MIN_GRAD_NORM),
                          'optimizer': QuasiNewtonOptimizer},
                     'sr1':
                         {'setting': QuasiNewtonOptSetting(x0=vK0,
                                                           step_method='sr1',
                                                           min_grad_norm=MIN_GRAD_NORM),
                          'optimizer': QuasiNewtonOptimizer}
                     }

    settings_dict_trustregion = {
                     'dogleg':
                         {'setting': TrustRegionOptSetting(x0=vK0,
                                                           step_method='dogleg',
                                                           min_grad_norm=MIN_GRAD_NORM),
                          'optimizer': TrustRegionOptimizer},
                     '2d_subspace':
                         {'setting': TrustRegionOptSetting(x0=vK0,
                                                           step_method='2d_subspace',
                                                           min_grad_norm=MIN_GRAD_NORM),
                          'optimizer': TrustRegionOptimizer},
                     'cg_steihaug':
                         {'setting': TrustRegionOptSetting(x0=vK0,
                                                           step_method='cg_steihaug',
                                                           min_grad_norm=MIN_GRAD_NORM),
                          'optimizer': TrustRegionOptimizer},
                    }

    settings_dp = {'policy_iteration': {'optimizer': policy_iteration},
                   'value_iteration': {'optimizer': value_iteration}}

    settings_dict_list = [settings_dict_gradient, settings_dict_linesearch, settings_dict_trustregion, settings_dp]
    categories = ['Gradient', 'Line search', 'Trust region', 'Dynamic programming']
    categories_results_dict = {}
    for settings_dict, category in zip(settings_dict_list, categories):
        results_dict = {}
        for method, method_dict in settings_dict.items():
            time_start = time()
            if category == 'Dynamic programming':
                # Get the optimizer
                optimizer = method_dict['optimizer']
                # Optimize
                print(method+' method')
                t_hist, x_hist, f_hist, g_hist, h_hist = optimizer(K0, A, B, Q, X0, min_grad_norm=MIN_GRAD_NORM)
            else:
                # Make the optimizer
                setting = method_dict['setting']
                Opt = method_dict['optimizer']
                opt = Opt(setting)

                # Optimize
                print(method+' method')
                t_hist, x_hist, f_hist, g_hist, h_hist = opt.optimize(obj, hidden_data=[A, B, Q, X0])

            time_end = time()
            # Store results
            result = {'t_hist': t_hist,
                      'x_hist': x_hist,
                      'f_hist': f_hist,
                      'g_hist': g_hist,
                      'h_hist': h_hist,
                      'elapsed_time': time_end - time_start}
            results_dict[method] = {'setting': setting,
                                    'result': result}

        # # Check optimality
        # for method, result_dict in results_dict.items():
        #     # Check if policy optimization solution satisfies the DARE
        #     x_hist = result['x_hist']
        #     K = mat(x_hist[-1], (m, n))
        #     check_are(K, A, B, Q)

        categories_results_dict[category] = results_dict

    plot_folder = 'plots'
    for category, results_dict in categories_results_dict.items():
        # # Plot each method in a separate figure
        # for method, result_dict in results_dict.items():
        #     result = result_dict['result']
        #     plot_single(result, x_opt=vec(K_are), f_opt=f_are, method=method)
        # Plot all methods together
        fig, ax = plot_multi(results_dict, x_opt=vec(K_are), f_opt=f_are, category=category)
        filename = 'demo_results_'+category.lower().replace(' ', '_')+'.pdf'
        path_out = os.path.join(plot_folder, filename)
        fig.savefig(path_out)

    # Show a representative collection of methods
    categories_elite = ['Gradient', 'Gradient', 'Line search', 'Line search', 'Line search', 'Trust region', 'Dynamic programming']
    methods_elite = ['gradient', 'relativistic', 'gradient (line search)', 'newton', 'bfgs', 'dogleg', 'policy_iteration']
    results_dict_elite = {}
    for category, method in zip(categories_elite, methods_elite):
        if method == 'gradient (line search)':
            old_method = 'gradient'
        else:
            old_method = method
        results_dict_elite[method] = categories_results_dict[category][old_method]
    fig, ax = plot_multi(results_dict_elite, x_opt=vec(K_are), f_opt=f_are, category='Representative')
    filename = 'demo_results_'+'representative'+'.pdf'
    path_out = os.path.join(plot_folder, filename)
    fig.savefig(path_out)
