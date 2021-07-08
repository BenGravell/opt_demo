import autograd.numpy as np
import autograd.numpy.linalg as la
from scipy.linalg import cho_factor, cho_solve, LinAlgError
from scipy.optimize import line_search as sp_line_search

from lqr_utility import quadratic_formula, posdefify, vec, mat, specrad
from colors import PrintColors


class Objective:
    def __init__(self, function, gradient, hessian, name=None):
        self.function = function
        self.gradient = gradient
        self.hessian = hessian
        self.name = name


class OptSetting:
    def __init__(self, x0, max_iters=None, min_grad_norm=None,
                 verbose=None, verbose_start=None, verbose_stride=None):
        self.x0 = x0

        if max_iters is None:
            max_iters = 100
        self.max_iters = max_iters

        if min_grad_norm is None:
            min_grad_norm = 1e-3
        self.min_grad_norm = min_grad_norm

        if verbose is None:
            verbose = True
        self.verbose = verbose

        if verbose_start is None:
            verbose_start = 20
        self.verbose_start = verbose_start

        if verbose_stride is None:
            verbose_stride = 10
        self.verbose_stride = verbose_stride


class GradientOptSetting(OptSetting):
    def __init__(self, x0, max_iters=None, min_grad_norm=None,
                 a0=None, step_method=None,
                 v0=None, mass=None, delta=None,  # momentum, nesterov, relativistic
                 avg_sq_grad0=None, gamma=None, eps=None,  # rmsprop
                 b1=None, b2=None, mean0=None, variance0=None,  # adam
                 verbose=None, verbose_start=None, verbose_stride=None):
        super().__init__(x0, max_iters, min_grad_norm, verbose, verbose_start, verbose_stride)

        if a0 is None:
            a0 = 1.0
        self.a0 = a0

        if step_method is None:
            step_method = 'gradient'
        self.step_method = step_method

        if v0 is None:
            v0 = np.zeros_like(x0)
        self.v0 = v0

        if mass is None:
            mass = 0.0
        self.mass = mass

        if delta is None:
            delta = 10.0
        self.delta = delta

        if avg_sq_grad0 is None:
            avg_sq_grad0 = np.ones_like(x0)
        self.avg_sq_grad0 = avg_sq_grad0

        if gamma is None:
            gamma = 0.9
        self.gamma = gamma

        if eps is None:
            eps = 1e-8
        self.eps = eps

        if b1 is None:
            b1 = 0.9
        self.b1 = b1

        if b2 is None:
            b2 = 0.999
        self.b2 = b2

        if mean0 is None:
            mean0 = np.zeros_like(x0)
        self.mean0 = mean0

        if variance0 is None:
            variance0 = np.zeros_like(x0)
        self.variance0 = variance0


class LineSearchOptSetting(OptSetting):
    def __init__(self, x0, max_iters=None, min_grad_norm=None,
                 a0=None, step_method=None, linesearch_method=None, pos_hess_eps=None,
                 verbose=None, verbose_start=None, verbose_stride=None):
        super().__init__(x0, max_iters, min_grad_norm, verbose, verbose_start, verbose_stride)

        if a0 is None:
            a0 = 1.0
        self.a0 = a0

        if step_method is None:
            step_method = 'gradient'
        self.step_method = step_method

        if linesearch_method is None:
            linesearch_method = 'strong_wolfe'
        self.linesearch_method = linesearch_method

        if pos_hess_eps is None:
            pos_hess_eps = 1e-6
        self.pos_hess_eps = pos_hess_eps


class QuasiNewtonOptSetting(LineSearchOptSetting):
    def __init__(self, x0, max_iters=None, min_grad_norm=None,
                 a0=None, step_method=None, step_length_method=None, pos_hess_eps=None,
                 H0=None, sr1_skip_tol=None,
                 verbose=None, verbose_start=None, verbose_stride=None):

        super().__init__(x0, max_iters, min_grad_norm,
                         a0, step_method, step_length_method, pos_hess_eps,
                         verbose, verbose_start, verbose_stride)

        if H0 is None:
            n = x0.size
            H0 = np.eye(n)
        self.H0 = H0

        if sr1_skip_tol is None:
            sr1_skip_tol = 1e-8
        self.sr1_skip_tol = sr1_skip_tol


class TrustRegionOptSetting(OptSetting):
    def __init__(self, x0, max_iters=None, min_grad_norm=None,
                 trust_radius0=1.0, trust_radius_max=10.0, step_method='dogleg', update_method='trust_region',
                 pos_hess_eps=1e-6,
                 verbose=None, verbose_start=None, verbose_stride=None):
        super().__init__(x0, max_iters, min_grad_norm, verbose, verbose_start, verbose_stride)

        self.trust_radius0 = trust_radius0
        self.trust_radius_max = trust_radius_max
        self.step_method = step_method
        self.update_method = update_method
        self.pos_hess_eps = pos_hess_eps


class Optimizer:
    def __init__(self, setting):
        self.setting = setting

    def update(self, x, obj, state_aux):
        raise NotImplementedError('Optimizer needs an update method!')

    def init_state_aux(self):
        raise NotImplementedError('Optimizer needs an init_state_aux method!')

    def optimize(self, obj, hidden_data=None):
        if hidden_data is not None:
            A, B, Q, X0 = hidden_data
            # n, m = B.shape

        def join_strings(word_list, display_width=16, spacer='    '):
            new_list = [f'{word:>{display_width}}' for word in word_list]
            return spacer.join(new_list)

        if self.setting.verbose:
            tags = []
            print_cols = ['iteration',
                          'objective_value',
                          'gradient_norm',
                          'hess_min',
                          'hess_max']
            header = join_strings(print_cols)
            print(header)

            def print_line(i, f, g, h):
                # Print per-iteration diagnostic info
                gi = la.norm(g)
                hi = np.sort(la.eig(h)[0])
                hi_min = hi[0]
                hi_max = hi[-1]
                current_cols = ['%d' % i,
                                '%.3e' % f,
                                '%.3e' % gi,
                                '%.3e' % hi_min,
                                '%.3e' % hi_max]
                line = join_strings(current_cols)
                if tags:
                    line = line+'    '+'    '.join(tags)
                print(line)
                return line

        # Initialize dimension, iterate, step length, state_aux quantities
        n = self.setting.x0.size
        x = np.copy(self.setting.x0)
        state_aux = self.init_state_aux()
        converged = False

        # Pre-allocate history arrays
        t_hist = np.arange(self.setting.max_iters)
        x_hist = np.zeros([self.setting.max_iters, n])
        f_hist = np.zeros(self.setting.max_iters)
        g_hist = np.zeros([self.setting.max_iters, n])
        h_hist = np.zeros([self.setting.max_iters, n, n])

        # Perform iterative optimization
        for i in range(self.setting.max_iters):
            # if hidden_data is not None:
            #     K = mat(x, (m, n))
            #     rho = specrad(A + np.dot(B, K))
            #     print(rho)

            # Record history
            f = obj.function(x)
            g = obj.gradient(x)
            h = obj.hessian(x)
            x_hist[i] = np.copy(x)
            f_hist[i] = np.copy(f)
            g_hist[i] = np.copy(g)
            h_hist[i] = np.copy(h)

            if self.setting.verbose:
                if (i <= self.setting.verbose_start) or (i % self.setting.verbose_stride == 0):
                    print_line(i, f, g, h)

            # Check if gradient has fallen below termination limit
            if la.norm(g) < self.setting.min_grad_norm:
                # Trim off unused part of history matrices
                t_hist = t_hist[0:i+1]
                x_hist = x_hist[0:i+1]
                f_hist = f_hist[0:i+1]
                g_hist = g_hist[0:i+1]
                h_hist = h_hist[0:i+1]
                converged = True
                break

            # Take a step to get the next iterate
            x, state_aux, tags = self.update(x, obj, state_aux)

        if not converged:
            # Record history
            f = obj.function(x)
            g = obj.gradient(x)
            h = obj.hessian(x)
            x_hist[-1] = np.copy(x)
            f_hist[-1] = np.copy(f)
            g_hist[-1] = np.copy(g)
            h_hist[-1] = np.copy(h)

        if self.setting.verbose:
            print_line(i+1, f, g, h)

            if converged:
                print(f"{PrintColors.OKGREEN}Optimization converged successfully!{PrintColors.ENDC}")
            else:
                print(f"{PrintColors.FAIL}Optimization failed to converge, stopping early!{PrintColors.ENDC}")

        print('')
        return t_hist, x_hist, f_hist, g_hist, h_hist


class GradientOptimizer(Optimizer):
    def __init__(self, setting):
        super().__init__(setting)

    def update(self, x, obj, state_aux):
        tags = []
        method = self.setting.step_method
        a = self.setting.a0
        if method not in ['nesterov', 'relativistic']:
            g = obj.gradient(x)

        if method == 'gradient':
            return x - a*g, None, tags
        elif method == 'momentum':
            x_old = state_aux
            mass = self.setting.mass
            y = x + mass*(x-x_old)
            return y - a*g, x, tags
        elif method == 'nesterov':
            x_old = state_aux
            mass = self.setting.mass
            y = x + mass*(x-x_old)
            gy = obj.gradient(y)
            return y - a*gy, x, tags
        elif method == 'relativistic':
            # See the paper https://arxiv.org/abs/1903.04100
            mass = self.setting.mass
            delta = self.setting.delta
            sqrtmass = mass**0.5
            v = state_aux
            x_pre = x + (sqrtmass/((mass*delta*np.sum(v**2) + 1)**0.5))*v
            v_pre = sqrtmass*v - a*obj.gradient(x_pre)
            x = x_pre + (1/(delta*np.sum(v_pre**2) + 1)**0.5)*v_pre
            v = sqrtmass*v_pre
            return x, v, tags
        elif method == 'rmsprop':
            # Root mean squared prop: See Adagrad paper for details.
            avg_sq_grad = state_aux
            gamma = self.setting.gamma
            eps = self.setting.eps
            avg_sq_grad = avg_sq_grad*gamma + (g**2)*(1-gamma)
            return x - a*g/(np.sqrt(avg_sq_grad)+eps), avg_sq_grad, tags
        elif method == 'adam':
            # Adam as described in http://arxiv.org/pdf/1412.6980.pdf.
            # Like RMSprop with momentum and some correction terms.
            mean, variance, i = state_aux
            eps = self.setting.eps
            b1 = self.setting.b1
            b2 = self.setting.b2
            mean = (1-b1) * g+b1 * mean  # First  moment estimate
            variance = (1-b2) * (g**2)+b2 * variance  # Second moment estimate
            mean_hat = mean / (1-b1**(i+1))  # Bias correction
            variance_hat = variance / (1-b2**(i+1))
            return x - a*mean_hat/(np.sqrt(variance_hat)+eps), [mean, variance, i+1], tags
        else:
            raise ValueError('Invalid step method chosen!')

    def init_state_aux(self):
        method = self.setting.step_method
        if method == 'gradient':
            return None
        elif method == 'momentum':
            v0 = np.copy(self.setting.v0)
            return v0
        elif method == 'nesterov':
            return self.setting.x0
        elif method == 'relativistic':
            v0 = np.copy(self.setting.v0)
            return v0
        elif method == 'rmsprop':
            avg_sq_grad0 = np.copy(self.setting.avg_sq_grad0)
            return avg_sq_grad0
        elif method == 'adam':
            mean0 = np.copy(self.setting.mean0)
            variance0 = np.copy(self.setting.variance0)
            i0 = 0
            return mean0, variance0, i0
        else:
            raise ValueError('Invalid step method chosen!')


class LineSearchOptimizer(Optimizer):
    def __init__(self, setting):
        super().__init__(setting)

    def calc_step_direction(self, x, obj, state_aux):
        method = self.setting.step_method
        if method == 'gradient':
            return -obj.gradient(x)
        elif method == 'newton':
            H = obj.hessian(x)
            B = posdefify(H, self.setting.pos_hess_eps)
            return -la.solve(B, obj.gradient(x))
        else:
            raise ValueError('Invalid step method!')

    def line_search(self, x, p, obj, a0=None, tol=1e-4, max_iter=1000, step_scale=0.9, curv_tol=0.9, method=None):
        if a0 is None:
            a0 = self.setting.a0
        a = np.copy(a0)

        if method is None:
            method = self.setting.linesearch_method

        # Check if p is a descent direction
        gp = np.dot(obj.gradient(x), p)
        if gp > 0:
            print(f"{PrintColors.WARNING}WARNING Search direction is NOT a descent direction, flipping sign {PrintColors.ENDC}")
            p = -p

        if method == 'backtrack':
            # Backtracking line search
            def sufficient_decrease(a):
                lhs = obj.function(x + a*p)
                rhs = obj.function(x) + tol*a*np.dot(obj.gradient(x), p)
                return lhs <= rhs
            i = 0
            while not sufficient_decrease(a):
                if i >= max_iter:
                    break
                a *= step_scale
                i += 1
        elif method == 'strong_wolfe':
            # Line search to satisfy strong Wolfe conditions
            if tol >= curv_tol:
                raise ValueError('Sufficient decrease tol (c1) must be less than curvature tol (c2)!')
            ls_result = sp_line_search(obj.function, obj.gradient, x, p, c1=tol, c2=curv_tol, amax=a0, maxiter=max_iter)
            a = ls_result[0]
            if a is None:
                print(f"{PrintColors.WARNING}WARNING Strong Wolfe line search failed to converge, resorting to backtracking{PrintColors.ENDC}")
                a = self.line_search(x, p, obj, a0, tol, max_iter=np.inf, step_scale=step_scale, curv_tol=curv_tol, method='backtrack')
        else:
            raise ValueError('Invalid line search method!')
        return a

    def update(self, x, obj, state_aux):
        tags = []
        # Construct the step direction and length
        p = self.calc_step_direction(x, obj, state_aux)
        a = self.line_search(x, p, obj)

        # Take a step to get the next iterate
        return x + a*p, state_aux, tags

    def init_state_aux(self):
        a0 = np.copy(self.setting.a0)
        return a0


class QuasiNewtonOptimizer(LineSearchOptimizer):
    def __init__(self, setting):
        super().__init__(setting)

    def calc_step_direction(self, x, obj, state_aux):
        a, Hinv = state_aux
        Hinv_pos = posdefify(Hinv, 0)
        return -np.dot(Hinv_pos, obj.gradient(x))

    def update_hessian_inverse(self, x, obj, p, state_aux):
        method = self.setting.step_method
        a, Hinv = state_aux
        n = x.size
        s = a*p
        y = obj.gradient(x + a*p) - obj.gradient(x)

        if method == 'bfgs':
            ssT = np.outer(s, s)
            ysT = np.outer(y, s)
            yTs = np.dot(y, s)
            C = np.eye(n) - ysT/yTs
            Hinv_new = np.dot(C.T, np.dot(Hinv, C)) + ssT/yTs
        elif method == 'dfp':
            Hinv_y = np.dot(Hinv, y)
            y_Hinv_y = np.dot(y, Hinv_y)
            ssT = np.outer(s, s)
            yTs = np.dot(y, s)
            Hinv_new = Hinv - np.outer(Hinv_y, Hinv_y)/y_Hinv_y + ssT/yTs
        elif method == 'sr1':
            Hinv_y = np.dot(Hinv, y)
            s_minus_Hinv_y = s - Hinv_y
            denominator = np.dot(s_minus_Hinv_y, y)
            if np.abs(denominator) > self.setting.sr1_skip_tol*la.norm(y)*la.norm(s_minus_Hinv_y):
                Hinv_new = Hinv + np.outer(s_minus_Hinv_y, s_minus_Hinv_y)/denominator
            else:  # skipping rule to avoid huge search directions under denominator collapse
                Hinv_new = np.copy(Hinv)
        else:
            raise ValueError('Invalid step method!')
        return Hinv_new

    def update(self, x, obj, state_aux):
        tags = []
        _, Hinv = state_aux
        # Construct the step direction and length
        p = self.calc_step_direction(x, obj, state_aux)
        a = self.line_search(x, p, obj)
        Hinv_new = self.update_hessian_inverse(x, obj, p, [a, Hinv])

        # Take a step to get the next iterate
        return x + a*p, [a, Hinv_new], tags

    def init_state_aux(self):
        a0 = np.copy(self.setting.a0)
        H0 = np.copy(self.setting.H0)
        return a0, H0


class TrustRegionOptimizer(Optimizer):
    def __init__(self, setting):
        super().__init__(setting)

    def model(self, x, p, obj):
        f = obj.function(x)
        g = obj.gradient(x)
        B = obj.hessian(x)
        return f + np.dot(g, p) + 0.5*np.dot(p, np.dot(B, p))

    def calc_step(self, x, trust_radius, obj):
        tags = []
        method = self.setting.step_method
        if method == 'dogleg':
            n = x.size
            g = obj.gradient(x)
            H = obj.hessian(x)
            B = posdefify(H, self.setting.pos_hess_eps)

            # Find the minimizing tau along the dogleg path
            pU = -(np.dot(g, g)/np.dot(g, np.dot(B, g)))*g
            pB = -la.solve(B, g)
            dp = pB - pU
            if la.norm(pB) <= trust_radius:
                # Minimum of model lies inside the trust region
                p = np.copy(pB)
            else:
                # Minimum of model lies outside the trust region
                tau_U = trust_radius/la.norm(pU)
                if tau_U <= 1:
                    # First dogleg segment intersects trust region boundary
                    p = tau_U*pU
                else:
                    # Second dogleg segment intersects trust region boundary
                    aa = np.dot(dp, dp)
                    ab = 2*np.dot(dp, pU)
                    ac = np.dot(pU, pU) - trust_radius**2
                    alphas = quadratic_formula(aa, ab, ac)
                    alpha = np.max(alphas)
                    p = pU + alpha*dp
            return p, tags

        elif method == '2d_subspace':
            g = obj.gradient(x)
            H = obj.hessian(x)
            B = posdefify(H, self.setting.pos_hess_eps)

            # Project g and B onto the 2D-subspace spanned by (normalized versions of) -g and -B^-1 g
            s1 = -g
            s2 = -la.solve(B, g)
            Sorig = np.vstack([s1, s2]).T
            S, Rtran = la.qr(Sorig)  # This is necessary for us to use same trust_radius before/after transforming
            g2 = np.dot(S.T, g)
            B2 = np.dot(S.T, np.dot(B, S))

            # Solve the 2D trust-region subproblem
            try:
                R, lower = cho_factor(B2)
                p2 = -cho_solve((R, lower), g2)
                p22 = np.dot(p2, p2)
                if np.dot(p2, p2) <= trust_radius**2:
                    p = np.dot(S, p2)
                    return p, tags
            except LinAlgError:
                pass

            a = B2[0, 0] * trust_radius**2
            b = B2[0, 1] * trust_radius**2
            c = B2[1, 1] * trust_radius**2

            d = g2[0] * trust_radius
            f = g2[1] * trust_radius

            coeffs = np.array([-b+d, 2*(a-c+f), 6*b, 2*(-a+c+f), -b-d])
            t = np.roots(coeffs)  # Can handle leading zeros
            t = np.real(t[np.isreal(t)])

            p2 = trust_radius * np.vstack((2*t/(1+t**2), (1-t**2)/(1+t**2)))
            value = 0.5 * np.sum(p2*np.dot(B2, p2), axis=0) + np.dot(g2, p2)
            i = np.argmin(value)
            p2 = p2[:, i]

            # Project back into the original n-dim space
            p = np.dot(S, p2)
            return p, tags

        elif method == 'cg_steihaug':
            # Settings
            max_iters = 100000  # TODO put in settings

            # Init
            n = x.size
            g = obj.gradient(x)
            B = obj.hessian(x)

            z = np.zeros(n)
            r = np.copy(g)
            d = -np.copy(g)

            # Choose eps according to Algo 7.1
            grad_norm = la.norm(g)
            eps = min(0.5, grad_norm**0.5)*grad_norm

            if la.norm(r) < eps:
                p = np.zeros(n)
                tags.append('Stopping tolerance reached!')
                return p, tags

            j = 0
            while j+1 < max_iters:
                # Check if 'd' is a direction of non-positive curvature
                dBd = np.dot(d, np.dot(B, d))
                rr = np.dot(r, r)
                if dBd <= 0:
                    ta = np.dot(d, d)
                    tb = 2*np.dot(d, z)
                    tc = np.dot(z, z) - trust_radius**2
                    taus = quadratic_formula(ta, tb, tc)
                    tau = np.max(taus)
                    p = z + tau*d
                    tags.append('Negative curvature encountered!')
                    return p, tags

                alpha = rr/dBd
                z_new = z + alpha*d

                # Check if trust region bound violated
                if la.norm(z_new) >= trust_radius:
                    ta = np.dot(d, d)
                    tb = 2*np.dot(d, z)
                    tc = np.dot(z, z) - trust_radius**2
                    taus = quadratic_formula(ta, tb, tc)
                    tau = np.max(taus)
                    p = z + tau*d
                    tags.append('Trust region boundary reached!')
                    return p, tags

                z = np.copy(z_new)
                r = r + alpha*np.dot(B, d)
                rr_new = np.dot(r, r)

                if la.norm(r) < eps:
                    p = np.copy(z)
                    tags.append('Stopping tolerance reached!')
                    return p, tags

                beta = rr_new/rr
                d = -r + beta*d

                j += 1

            p = np.zeros(n)
            tags.append('ALERT!  CG-Steihaug failed to solve trust-region subproblem within max_iters')
            return p, tags
        else:
            raise ValueError('Invalid step method!')

    def calc_update(self, x, p, trust_radius, trust_radius_max, obj,
                    quality_required=0.2, quality_low=0.25, quality_high=0.75):
        # Parameter checks
        if not quality_required < quality_low < quality_high:
            raise ValueError('Invalid quality parameters, must be: quality_required < quality_low < quality_high')

        df = obj.function(x) - obj.function(x + p)
        dm = self.model(x, np.zeros_like(x), obj) - self.model(x, p, obj)
        quality = df/dm

        if quality < quality_low:
            trust_radius_new = quality_low*trust_radius
        else:
            if quality > quality_high and np.isclose(la.norm(p), trust_radius):
                trust_radius_new = min(2*trust_radius, trust_radius_max)
            else:
                trust_radius_new = np.copy(trust_radius)

        if quality > quality_required:
            x_new = x + p
        else:
            x_new = np.copy(x)

        return x_new, trust_radius_new

    def update(self, x, obj, state_aux):
        trust_radius, trust_radius_max = state_aux
        p, tags = self.calc_step(x, trust_radius, obj)
        x, trust_radius = self.calc_update(x, p, trust_radius, trust_radius_max, obj)
        return x, [trust_radius, trust_radius_max], tags

    def init_state_aux(self):
        trust_radius = np.copy(self.setting.trust_radius0)
        trust_radius_max = np.copy(self.setting.trust_radius_max)
        return [trust_radius, trust_radius_max]
