import torch
from einops import rearrange
from functorch import vmap
from geomstats.geometry.special_orthogonal import SpecialOrthogonal

from flowmatch.utils.so3_helpers import expmap, norm_SO3, geodesic_distance, rotmat_to_rotvec

def riemannian_gradient(f, R):
    coefficients = torch.zeros(list(R.shape[:-2])+[3], requires_grad=True).to(R.device)
    R_delta  = expmap(R, R @ hat(coefficients))
    grad_coefficients = torch.autograd.grad(f(R_delta).sum(), coefficients, )[0]
    return R @ hat(grad_coefficients)


class SO3ConditionalFlowMatcher:
    def __init__(self, manifold):
        self.sigma = None
        self.manifold = manifold
        self.vec_manifold = SpecialOrthogonal(n=3, point_type="vector")

    def vec_log_map(self, x0, x1):
        # get logmap of x_1 from x_0
        # convert to axis angle to compute logmap efficiently
        rot_x0 = rotmat_to_rotvec(x0)
        rot_x1 = rotmat_to_rotvec(x1)

        torch.set_default_dtype(torch.float64)
        log_x1 = self.vec_manifold.log_not_from_identity(rot_x1, rot_x0)
        torch.set_default_dtype(torch.float32)
        return log_x1, rot_x0

    def sample_xt(self, x0, x1, t):
        # sample along the geodesic from x0 to x1
        log_x1, rot_x0 = self.vec_log_map(x0.double(), x1.double())
        # group exponential at x0
        torch.set_default_dtype(torch.float64)
        xt = self.vec_manifold.exp_not_from_identity(t.reshape(-1, 1) * log_x1, rot_x0)
        xt = self.vec_manifold.matrix_from_rotation_vector(xt)
        torch.set_default_dtype(torch.float32)
        xt = torch.from_numpy(xt).double()
        return xt

    def compute_conditional_flow_simple(self, t, xt):
        xt = rearrange(xt, "b c d -> b (c d)", c=3, d=3)

        def index_time_der(i):
            return torch.autograd.grad(xt, t, i, create_graph=True, retain_graph=True)[
                0
            ]

        xt_dot = vmap(index_time_der, in_dims=1)(
            torch.eye(9).to(xt.device).repeat(xt.shape[0], 1, 1)
        )
        return rearrange(xt_dot, "(c d) b -> b c d", c=3, d=3)
        

    def compute_conditional_flow(self, x_0, x_1, t, x_t):
        # compute the geodesic distance
        dist_x0_x1 = geodesic_distance(x0, x1)                       # d(x0, x1)
        geo_dist = lambda x: geodesic_distance(x, x1)
        dist_grad_wrt_xt = riemannian_gradient(geo_dist, xt) # nabla_xt d(xt, x1)

        # Compute the geodesic norm ||.||_g:
        denom_term = norm_SO3(xt, dist_grad_wrt_xt)

        output = -dist_x0_x1[:, None, None] * dist_grad_wrt_xt / denom_term[:, None, None]
        return output
        

    def sample_location_and_conditional_flow_simple(self, x0, x1):
        t = torch.rand(x0.shape[0]).type_as(x0).to(x0.device)
        t.requires_grad = True
        xt = self.sample_xt(x0, x1, t)
        ut = self.compute_conditional_flow_simple(t, xt)

        return t, xt, ut
