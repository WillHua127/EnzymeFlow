import logging
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from functorch import vmap
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from scipy.spatial.transform import Rotation
from torch import Tensor


from flowmatch.utils.so3_helpers import (
    exp,
    expmap,
    hat,
    log,
)

from flowmatch.utils.rigid_helpers import (
    assemble_rigid_mat,
    extract_trans_rots_mat,
)

from flowmatch.utils.condflowmatcher import ConditionalFlowMatcher
from flowmatch.utils.so3_condflowmatcher import SO3ConditionalFlowMatcher

from ofold.utils import rigid_utils as ru


class SO3FM:
    def __init__(self, so3_conf):
        self._log = logging.getLogger(__name__)
        self.so3_group = SpecialOrthogonal(n=3, point_type="matrix")
        self.so3_cfm = SO3ConditionalFlowMatcher(manifold=self.so3_group)
        self.g = so3_conf.g
        self.min_sigma = so3_conf.min_sigma
        self.inference_scaling = so3_conf.inference_scaling

    def sample(self, t: float, n_samples: float = 1):
        return Rotation.random(n_samples).as_matrix()

    def sample_ref(self, n_samples: float = 1):
        return self.sample(1, n_samples=n_samples)

    def compute_sigma_t(self, t):
        if isinstance(t, float):
            t = torch.tensor(t)
        return torch.sqrt(self.g**2 * t * (1 - t) + self.min_sigma**2)

    def forward_marginal(
        self, rot_1: np.ndarray, t: float, rot_0: Union[torch.Tensor, None] = None, time_der=True,
    ):
        seq_len = rot_1.shape[1]
        n_samples = len(rot_1)

        # Sample Unif w.r.t Hfeatr Measure on SO(3)
        # This corresponds IGSO(3) with high concentration param
        rot_0 = self.sample_ref(n_samples) if rot_0 is None else rot_0
        t = torch.tensor(t).repeat(rot_1.shape[0])
        rot_1 = torch.from_numpy(rot_1).double()
        rot_0 = torch.from_numpy(rot_0).double()
        rot_t = self.geodesic_t(t[..., None], rot_1, rot_0)

        return rot_t

    def reverse_euler(
        self,
        rot_1: np.ndarray,
        rot_t: np.ndarray,
        t: float,
        dt: float,
        flow_mask: np.ndarray = None,
        noise_scale: float = 1.0,
        sample_schedule = 'exp',
        exp_rate = 10,
    ):
        if not np.isscalar(t):
            raise ValueError(f"{t} must be a scalar.")

        if sample_schedule == 'linear':
            scaling = 1 / (1 - t)
        elif sample_schedule == 'exp':
            scaling = exp_rate

        rot_t_1 = self.geodesic_t(scaling * dt, torch.tensor(rot_1).double(), torch.tensor(rot_t).double())
        return rot_t_1.detach().cpu().numpy()
        

    def reverse(
        self,
        rot_t: np.ndarray,
        v_t: np.ndarray,
        t: float,
        dt: float,
        flow_mask: np.ndarray = None,
        noise_scale: float = 1.0,
    ):
        if not np.isscalar(t):
            raise ValueError(f"{t} must be a scalar.")

        perturb = v_t * dt

        if flow_mask is not None:
            perturb *= flow_mask[..., None]

        rot_t_1 = expmap(torch.tensor(rot_t).double(), torch.tensor(perturb).double())
        rot_t_1 = rot_t_1.reshape(rot_t.shape)
        return rot_t_1.detach().cpu().numpy()


    def skew_matrix_exponential_map(self, angles: torch.Tensor, skew_matrices: torch.Tensor, tol=1e-10):
        # Set up identity matrix and broadcast.
        id3 = self._broadcast_identity(skew_matrices)
    
        # Broadcast angles and pre-compute square.
        angles = angles[..., None, None]
        angles_sq = angles.square()
    
        # Get standard terms.
        sin_coeff = torch.sin(angles) / angles
        cos_coeff = (1.0 - torch.cos(angles)) / angles_sq
        # Use second order Taylor expansion for values close to zero.
        sin_coeff_small = 1.0 - angles_sq / 6.0
        cos_coeff_small = 0.5 - angles_sq / 24.0
    
        mask_zero = torch.abs(angles) < tol
        sin_coeff = torch.where(mask_zero, sin_coeff_small, sin_coeff)
        cos_coeff = torch.where(mask_zero, cos_coeff_small, cos_coeff)
    
        # Compute matrix exponential using Rodrigues' formula.
        exp_skew = (
            id3
            + sin_coeff * skew_matrices
            + cos_coeff * torch.einsum("b...ik,b...kj->b...ij", skew_matrices, skew_matrices)
        )
        return exp_skew
        

    def geodesic_t(self, t: float, mat: torch.Tensor, base_mat: torch.Tensor, rot_vf=None):
        if rot_vf is None:
            rot_vf = self.vectorfield(base_mat, mat, t)
        mat_t = self.rotvec_to_rotmat(t * rot_vf)
        if base_mat.shape != mat_t.shape:
            raise ValueError(
                f'Incompatible shapes: base_mat={base_mat.shape}, mat_t={mat_t.shape}')
        return torch.einsum("...ij,...jk->...ik", base_mat.double(), mat_t.double())

    def vector_to_skew_matrix(self, vectors: torch.Tensor):
        # Generate empty skew matrices.
        skew_matrices = torch.zeros((*vectors.shape, 3), device=vectors.device, dtype=vectors.dtype)
    
        # Populate positive values.
        skew_matrices[..., 2, 1] = vectors[..., 0]
        skew_matrices[..., 0, 2] = vectors[..., 1]
        skew_matrices[..., 1, 0] = vectors[..., 2]
    
        # Generate skew symmetry.
        skew_matrices = skew_matrices - skew_matrices.transpose(-2, -1)
    
        return skew_matrices

    def rotvec_to_rotmat(self, rotation_vectors: torch.Tensor, tol: float = 1e-10):
        # Compute rotation angle as vector norm.
        rotation_angles = torch.norm(rotation_vectors, dim=-1)
    
        # Map axis to skew matrix basis.
        skew_matrices = self.vector_to_skew_matrix(rotation_vectors)
    
        # Compute rotation matrices via matrix exponential.
        rotation_matrices = self.skew_matrix_exponential_map(rotation_angles, skew_matrices, tol=tol)
    
        return rotation_matrices
        

    def _broadcast_identity(self, target: torch.Tensor):
        id3 = torch.eye(3, device=target.device, dtype=target.dtype)
        id3 = torch.broadcast_to(id3, target.shape)
        return id3


    def skew_matrix_to_vector(self, skew_matrices: torch.Tensor):
        vectors = torch.zeros_like(skew_matrices[..., 0])
        vectors[..., 0] = skew_matrices[..., 2, 1]
        vectors[..., 1] = skew_matrices[..., 0, 2]
        vectors[..., 2] = skew_matrices[..., 1, 0]
        return vectors


    def angle_from_rotmat(self, rotation_matrices: torch.Tensor):
        # Compute sine of angles (uses the relation that the unnormalized skew vector generated by a
        # rotation matrix has the length 2*sin(theta))
        skew_matrices = rotation_matrices - rotation_matrices.transpose(-2, -1)
        skew_vectors = self.skew_matrix_to_vector(skew_matrices)
        angles_sin = torch.norm(skew_vectors, dim=-1) / 2.0
        # Compute the cosine of the angle using the relation cos theta = 1/2 * (Tr[R] - 1)
        angles_cos = (torch.einsum("...ii", rotation_matrices) - 1.0) / 2.0
    
        # Compute angles using the more stable atan2
        angles = torch.atan2(angles_sin, angles_cos)
    
        return angles, angles_sin, angles_cos


    def rotmat_to_rotvec(self, rotation_matrices: torch.Tensor):
        # Get angles and sin/cos from rotation matrix.
        angles, angles_sin, _ = self.angle_from_rotmat(rotation_matrices)
        # Compute skew matrix representation and extract so(3) vector components.
        vector = self.skew_matrix_to_vector(rotation_matrices - rotation_matrices.transpose(-2, -1))
    
        # Three main cases for angle theta, which are captured
        # 1) Angle is 0 or close to zero -> use Taylor series for small values / return 0 vector.
        mask_zero = torch.isclose(angles, torch.zeros_like(angles)).to(angles.dtype)
        # 2) Angle is close to pi -> use outer product relation.
        mask_pi = torch.isclose(angles, torch.full_like(angles, np.pi), atol=1e-2).to(angles.dtype)
        # 3) Angle is unproblematic -> use the standard formula.
        mask_else = (1 - mask_zero) * (1 - mask_pi)
    
        # Compute case dependent pre-factor (1/2 for angle close to 0, angle otherwise).
        numerator = mask_zero / 2.0 + angles * mask_else
        # The Taylor expansion used here is actually the inverse of the Taylor expansion of the inverted
        # fraction sin(x) / x which gives better accuracy over a wider range (hence the minus and
        # position in denominator).
        denominator = (
            (1.0 - angles**2 / 6.0) * mask_zero  # Taylor expansion for small angles.
            + 2.0 * angles_sin * mask_else  # Standard formula.
            + mask_pi  # Avoid zero division at angle == pi.
        )
        prefactor = numerator / denominator
        vector = vector * prefactor[..., None]
    
        # For angles close to pi, derive vectors from their outer product (ww' = 1 + R).
        id3 = self._broadcast_identity(rotation_matrices)
        skew_outer = (id3 + rotation_matrices) / 2.0
        # Ensure diagonal is >= 0 for square root (uses identity for masking).
        skew_outer = skew_outer + (torch.relu(skew_outer) - skew_outer) * id3
    
        # Get basic rotation vector as sqrt of diagonal (is unit vector).
        vector_pi = torch.sqrt(torch.diagonal(torch.clamp(skew_outer, min=1e-8), dim1=-2, dim2=-1))
    
        # Compute the signs of vector elements (up to a global phase).
        # Fist select indices for outer product slices with the largest norm.
        signs_line_idx = torch.argmax(torch.norm(skew_outer, dim=-1), dim=-1).long()
        # Select rows of outer product and determine signs.
        signs_line = torch.take_along_dim(skew_outer, dim=-2, indices=signs_line_idx[..., None, None])
        signs_line = signs_line.squeeze(-2)
        signs = torch.sign(signs_line)
    
        # Apply signs and rotation vector.
        vector_pi = vector_pi * angles[..., None] * signs
    
        # Fill entries for angle == pi in rotation vector (basic vector has zero entries at this point).
        vector = vector + vector_pi * mask_pi[..., None]
    
        return vector

    def rot_transpose(self, mat: torch.Tensor):
        return torch.transpose(mat, -1, -2)


    def rot_mult(self, mat_1: torch.Tensor, mat_2: torch.Tensor):
        return torch.einsum("...ij,...jk->...ik", mat_1.double(), mat_2.double())

    def vectorfield(self, rot_t, rot_1, t):
        u_t = self.rotmat_to_rotvec(self.rot_mult(self.rot_transpose(rot_t), rot_1))
        return u_t

    def vectorfield_scaling(self, t: np.ndarray):
        return 1




###########################################
class R3FM:
    def __init__(self, r3_conf):
        self._r3_conf = r3_conf
        self.min_b = r3_conf.min_b
        self.max_b = r3_conf.max_b
        self.r3_cfm = ConditionalFlowMatcher()
        self.g = r3_conf.g
        self.min_sigma = r3_conf.min_sigma

    def _scale(self, x):
        return x * self._r3_conf.coordinate_scaling

    def _unscale(self, x):
        return x / self._r3_conf.coordinate_scaling

    def b_t(self, t):
        if np.any(t < 0) or np.any(t > 1):
            raise ValueError(f"Invalid t={t}")
        return self.min_b + t * (self.max_b - self.min_b)

    def drift_coef(self, x, t):
        """Time-dependent drift coefficient."""
        return -1 / 2 * self.b_t(t) * x

    def sample_ref(self, n_samples: float = 1):
        noise = np.random.normal(size=(n_samples, 3))
        return noise - noise.mean(0)

    def marginal_b_t(self, t):
        return t * self.min_b + (1 / 2) * (t**2) * (self.max_b - self.min_b)

    def calc_trans_0(self, score_t, x_t, t, use_torch=True):
        beta_t = self.marginal_b_t(t)
        beta_t = beta_t[..., None, None]
        exp_fn = torch.exp if use_torch else np.exp
        cond_var = 1 - exp_fn(-beta_t)
        return (score_t * cond_var + x_t) / exp_fn(-1 / 2 * beta_t)

    def forward(self, x_t_1: np.ndarray, t: float, num_t: int):
        if not np.isscalar(t):
            raise ValueError(f"{t} must be a scalar.")
        x_t_1 = self._scale(x_t_1)
        b_t = torch.tensor(self.marginal_b_t(t) / num_t).to(x_t_1.device)
        z_t_1 = torch.tensor(np.random.normal(size=x_t_1.shape)).to(x_t_1.device)
        x_t = torch.sqrt(1 - b_t) * x_t_1 + torch.sqrt(b_t) * z_t_1
        return x_t

    def compute_sigma_t(self, t):
        if isinstance(t, float):
            t = torch.tensor(t)
        return torch.sqrt(self.g**2 * t * (1 - t) + self.min_sigma**2)

    
    def reverse_euler(
        self,
        *,
        x_1: np.ndarray,
        x_t: np.ndarray,
        t: float,
        dt: float,
        mask: np.ndarray = None,
        center: bool = True,
        noise_scale: float = 1.0,
        center_of_mass = None,
        sample_schedule = 'linear',
    ):
        if not np.isscalar(t):
            raise ValueError(f"{t} must be a scalar.")

        x_vf = (x_1 - x_t) / (1 - t)

        if mask is not None:
            x_vf *= mask[..., None]
        else:
            mask = np.ones(x_t.shape[:-1])
            
        x_t_1 = x_t + x_vf * dt
        if center and (center_of_mass is None):
            com = np.sum(x_t_1, axis=-2) / np.sum(mask, axis=-1)[..., None]
            x_t_1 = x_t_1 - com[..., None, :]
        
        elif center and (center_of_mass is not None):
            x_t_1 = x_t_1 - center_of_mass[..., None, :]
        
        return x_t_1

    
    def reverse(
        self,
        *,
        x_t: np.ndarray,
        v_t: np.ndarray,
        t: float,
        dt: float,
        mask: np.ndarray = None,
        center: bool = True,
        noise_scale: float = 1.0,
        center_of_mass = None,
    ):
        if not np.isscalar(t):
            raise ValueError(f"{t} must be a scalar.")
        x_t = self._scale(x_t)
        perturb = v_t * dt

        if mask is not None:
            perturb *= mask[..., None]
        else:
            mask = np.ones(x_t.shape[:-1])
            
        x_t_1 = x_t + perturb
        if center and (center_of_mass is None):
            com = np.sum(x_t_1, axis=-2) / np.sum(mask, axis=-1)[..., None]
            x_t_1 = x_t_1 - com[..., None, :]
        
        elif center and (center_of_mass is not None):
            x_t_1 = x_t_1 - center_of_mass[..., None, :]
        
        x_t_1 = self._unscale(x_t_1)
        return x_t_1

    def vectorfield_scaling(self, t: float):
        return 1

    def vectorfield(self, x_t, x_0, t, use_torch=False, scale=False):
        if scale:
            x_t = self._scale(x_t)
            x_0 = self._scale(x_0)
        return (x_t - x_0) / (t + 1e-10)
        


###########################################
class SE3FlowMatcher:
    def __init__(self, se3_conf):
        self._log = logging.getLogger(__name__)
        self._se3_conf = se3_conf
        self._do_fm_rot = se3_conf.flow_rot
        self._so3_fm = SO3FM(self._se3_conf.so3)
        self._flow_trans = se3_conf.flow_trans
        self._r3_fm = R3FM(self._se3_conf.r3)

        if se3_conf.ot_plan:
            self._log.info(f"Using OT plan with {self._se3_conf.ot_fn} computation.")

    def sample_multinomial(
        self,
        feat_t: torch.tensor,
        flow_mask: torch.tensor = None,
    ):
        feat_t = feat_t + 1e-10
        feat = torch.multinomial(feat_t, num_samples=1)
        if flow_mask is not None:
            feat = feat * flow_mask
            
        return feat

    
    def sample_masking(
        self,
        feat_0: torch.tensor,
        feat_1: torch.tensor,
        t: float,
        n_token = 20,
        flow_mask: torch.tensor = None,
    ):
        num_res = feat_1.size(0)
        u = torch.rand(num_res, device=feat_1.device)
        feat_multi_0 = torch.multinomial(feat_0, num_samples=1)
        feat_multi_t = feat_1.clone()
        corruption_mask = u < (1-t) # (N,)
        uniform_sample = torch.randint_like(feat_multi_t, low=0, high=n_token)
        feat_multi_t[corruption_mask] = uniform_sample[corruption_mask]
        
        if flow_mask is not None:
            feat_multi_t = feat_multi_t * flow_mask
            
        return feat_multi_t
    
    def sample_argmax(
        self,
        feat_t: torch.tensor,
        flow_mask: torch.tensor = None,
    ):
        feat_t = feat_t + 1e-10
        feat = feat_t.argmax(dim=-1, keepdim=True)
        if flow_mask is not None:
            feat = feat * flow_mask
            
        return feat

    def forward_masking(
        self,
        feat_0: torch.tensor,
        feat_1: torch.tensor,
        t: float,
        mask_token_idx: int,
        flow_mask: torch.tensor = None,
    ):
        if feat_1 is None:
            feat_t = torch.full(size=feat_0.size(), fill_value=mask_token_idx)

        else:
            feat_t = feat_1.clone()
            corruption_mask = feat_0 < (1 - t)
            feat_t[corruption_mask] = mask_token_idx

        if flow_mask is not None:
            feat_t = feat_t * flow_mask[..., None]
            
        return feat_t

    def forward_multinomial(
        self,
        feat_0: torch.tensor,
        feat_1: torch.tensor,
        t: float,
        flow_mask: torch.tensor = None,
    ):
        feat_0 = torch.multinomial(feat_0, num_samples=1).squeeze()

        if feat_1 is None:
            feat_t = feat_0.clone()

        else:
            u = torch.rand(size=feat_0.size())
            feat_t = feat_1.clone()
            corruption_mask = u < (1 - t)
            feat_t[corruption_mask] = feat_0[corruption_mask]

        if flow_mask is not None:
            feat_t = feat_t * flow_mask[..., None]
            
        return feat_t
    
    
    def reverse_masking_euler_purity(
        self,
        feat_t: torch.tensor,
        feat: torch.tensor,
        t: float,
        dt: float,
        n_token: int,
        mask_token_idx: int,
        flow_mask: torch.tensor = None,
        noise: float = 0.0,
        temp = 0.1,
    ):
        if feat.dim() == 2:            
            n_batch = feat.size(0)

            logits_1_wo_mask = feat[:, 0:-1] # (B, D, S-1)
            pt_x1_probs = F.softmax(logits_1_wo_mask / temp, dim=-1) # (B, S-1)
            max_logprob = torch.max(torch.log(pt_x1_probs), dim=-1)[0].reshape(n_batch, 1) # (B)
            max_logprob = max_logprob - (feat_t != mask_token_idx).float() * 1e10
            sorted_max_logprobs_idcs = torch.argsort(max_logprob, dim=-1, descending=True) # (B)

            unmask_probs = torch.tensor(dt * ((1 + noise * t) / (1-t))).float().to(feat_t.device).clamp(max=1) # scalar
            number_to_unmask = torch.binomial(count=torch.count_nonzero(feat_t == mask_token_idx, dim=-1).float(), prob=unmask_probs)
            unmasked_samples = torch.multinomial(pt_x1_probs.reshape(-1, n_token-1), num_samples=1).reshape(n_batch, 1)

            D_grid = torch.arange(1, device=feat_t.device).reshape(1, -1).repeat(n_batch, 1)
            mask1 = (D_grid < number_to_unmask.reshape(-1, 1)).float()

            inital_val_max_logprob_idcs = sorted_max_logprobs_idcs.reshape(-1, 1)
            masked_sorted_max_logprobs_idcs = (mask1 * sorted_max_logprobs_idcs + (1-mask1) * inital_val_max_logprob_idcs).long()
            mask2 = torch.zeros((n_batch, 1), device=feat_t.device)
            mask2.scatter_(dim=1, index=masked_sorted_max_logprobs_idcs, src=torch.ones((n_batch, 1), device=feat_t.device))
            unmask_zero_row = (number_to_unmask == 0).reshape(-1, 1).repeat(1, 1).float()
            mask2 = mask2 * (1 - unmask_zero_row)
            feat_t = feat_t * (1 - mask2) + unmasked_samples * mask2
    
            # re-mask
            u = torch.rand(n_batch, 1, device=feat_t.device)
            re_mask_mask = (u < dt * noise).float()
            feat_t_1 = feat_t * (1 - re_mask_mask) + mask_token_idx * re_mask_mask

            
        elif feat.dim() == 3:
            msa_feat_size = False
            if feat_t.size(1) == 1:
                msa_feat_size = True
                feat_t = feat_t.squeeze(dim=1)
            n_batch, n_res = feat.size(0), feat.size(1)

            logits_1_wo_mask = feat[:, :, 0:-1] # (B, D, S-1)
            pt_x1_probs = F.softmax(logits_1_wo_mask / temp, dim=-1) # (B, D, S-1)
            max_logprob = torch.max(torch.log(pt_x1_probs), dim=-1)[0] # (B, D)
            max_logprob = max_logprob - (feat_t != mask_token_idx).float() * 1e10
            sorted_max_logprobs_idcs = torch.argsort(max_logprob, dim=-1, descending=True) # (B, D)

            unmask_probs = torch.tensor(dt * ((1 + noise * t) / (1-t))).float().to(feat_t.device).clamp(max=1) # scalar
            number_to_unmask = torch.binomial(count=torch.count_nonzero(feat_t == mask_token_idx, dim=-1).float(), prob=unmask_probs)
            unmasked_samples = torch.multinomial(pt_x1_probs.reshape(-1, n_token-1), num_samples=1).reshape(n_batch, n_res)
        
            
            D_grid = torch.arange(n_res, device=feat_t.device).reshape(1, -1).repeat(n_batch, 1)
            mask1 = (D_grid < number_to_unmask.reshape(-1, 1)).float()
            inital_val_max_logprob_idcs = sorted_max_logprobs_idcs[:, 0].reshape(-1, 1).repeat(1, n_res)
            masked_sorted_max_logprobs_idcs = (mask1 * sorted_max_logprobs_idcs + (1-mask1) * inital_val_max_logprob_idcs).long()
            mask2 = torch.zeros((n_batch, n_res), device=feat_t.device)
            mask2.scatter_(dim=1, index=masked_sorted_max_logprobs_idcs, src=torch.ones((n_batch, n_res), device=feat_t.device))
            unmask_zero_row = (number_to_unmask == 0).reshape(-1, 1).repeat(1, n_res).float()
            mask2 = mask2 * (1 - unmask_zero_row)
            feat_t = feat_t * (1 - mask2) + unmasked_samples * mask2
    
            # re-mask
            u = torch.rand(n_batch, n_res, device=feat_t.device)
            re_mask_mask = (u < dt * noise).float()
            feat_t_1 = feat_t * (1 - re_mask_mask) + mask_token_idx * re_mask_mask

            if msa_feat_size:
                feat_t_1 = feat_t_1.unsqueeze(dim=1)
    
        return feat_t_1.long()               
        

    def reverse_masking_euler(
        self,
        feat_t: torch.tensor,
        feat: torch.tensor,
        t: float,
        dt: float,
        n_token: int,
        mask_token_idx: int,
        flow_mask: torch.tensor = None,
        noise: float = 0.0,
        temp = 0.1,
    ):
        if feat.dim() == 2:
            n_batch = feat.size(0)

            mask_one_hot = torch.zeros((n_token,), device=feat_t.device)
            mask_one_hot[mask_token_idx] = 1.0
            feat[..., mask_token_idx] = -1e10
            pt_x1_probs = F.softmax(feat / temp, dim=-1) # (B, D, S)

            feat_t_is_mask = (feat_t == mask_token_idx).view(n_batch, 1).float()
            step_probs = dt * pt_x1_probs * ((1+noise*t) / ((1 - t)))
            step_probs += dt * (1 - feat_t_is_mask) * mask_one_hot.view(1, -1) * noise

            step_probs = torch.clamp(step_probs, min=0.0, max=1.0)
            step_probs[
                torch.arange(n_batch, device=feat_t.device),
                feat_t.long().flatten()
            ] = 0.0
            step_probs[
                torch.arange(n_batch, device=feat_t.device),
                feat_t.long().flatten()
            ] = 1.0 - torch.sum(step_probs, dim=-1).flatten()
            step_probs = torch.clamp(step_probs, min=0.0, max=1.0)

            
        elif feat.dim() == 3:
            n_batch, n_res = feat.size(0), feat.size(1)
    
            mask_one_hot = torch.zeros((n_token,), device=feat_t.device)
            mask_one_hot[mask_token_idx] = 1.0
            feat[..., mask_token_idx] = -1e10
            pt_x1_probs = F.softmax(feat / temp, dim=-1) # (B, D, S)
            
            feat_t_is_mask = (feat_t == mask_token_idx).view(n_batch, n_res, 1).float()
            step_probs = dt * pt_x1_probs * ((1+noise*t) / ((1 - t))) # (B, D, S)
            step_probs += dt * (1 - feat_t_is_mask) * mask_one_hot.view(1, 1, -1) * noise
    
            step_probs = torch.clamp(step_probs, min=0.0, max=1.0)
            step_probs[
                torch.arange(n_batch, device=feat_t.device).repeat_interleave(n_res),
                torch.arange(n_res, device=feat_t.device).repeat(n_batch),
                feat_t.long().flatten()
            ] = 0.0
            step_probs[
                torch.arange(n_batch, device=feat_t.device).repeat_interleave(n_res),
                torch.arange(n_res, device=feat_t.device).repeat(n_batch),
                feat_t.long().flatten()
            ] = 1.0 - torch.sum(step_probs, dim=-1).flatten()
            step_probs = torch.clamp(step_probs, min=0.0, max=1.0)
    
            if flow_mask is not None:
                step_probs = step_probs * flow_mask[..., None]
    
    
        feat_t_1 = torch.multinomial(step_probs.view(-1, n_token), num_samples=1).view(feat_t.size())
        return feat_t_1
    

    def reverse_multinomial_euler(
        self,
        feat_t: torch.tensor,
        feat: torch.tensor,
        t: float,
        dt: float,
        n_token: int,
        flow_mask: torch.tensor = None,
        sample_multinomial = True,
        noise: float = 0.0,
    ):
        n_batch, n_res = feat_t.size(0), feat_t.size(1)

        pt_x1_probs = F.softmax(feat, dim=-1)
        pt_x1_eq_xt_prob = torch.gather(pt_x1_probs, dim=-1, index=feat_t.long().unsqueeze(-1))
        step_probs = dt * (pt_x1_probs * ((1 + noise + noise * (n_token - 1) * t) / (1-t)) + noise * pt_x1_eq_xt_prob)
        
        step_probs = torch.clamp(step_probs, min=0., max=1.)
        step_probs[
            torch.arange(n_batch, device=feat_t.device).repeat_interleave(n_res),
            torch.arange(n_res, device=feat_t.device).repeat(n_batch),
            feat_t.long().flatten()
        ] = 0.0
        step_probs[
            torch.arange(n_batch, device=feat_t.device).repeat_interleave(n_res),
            torch.arange(n_res, device=feat_t.device).repeat(n_batch),
            feat_t.long().flatten()
        ] = 1.0 - torch.sum(step_probs, dim=-1).flatten()
        step_probs = torch.clamp(step_probs, min=0.0, max=1.0)

        if flow_mask is not None:
            step_probs = step_probs * flow_mask[..., None]
            flow_mask = flow_mask.view(n_batch*n_res, -1)
        
        feat_t_1 = torch.multinomial(step_probs.view(-1, n_token), num_samples=1).view(feat_t.size())

        return feat_t_1
        

    def reverse_multinomial(
        self,
        feat_t: torch.tensor,
        feat_vectorfield: torch.tensor,
        t: float,
        dt: float,
        flow_mask: torch.tensor = None,
        sample_multinomial = True,
    ):
        if feat_t.dim() <= 2:
            n_batch, n_res = feat_t.size(0), 1
        else:
            n_batch, n_res = feat_t.size(0), feat_t.size(1)
            
        perturb = feat_vectorfield * dt
        feat_t_1 = feat_t + perturb
        #feat_t_1 = feat_t_1 / (feat_t_1.sum(dim=-1, keepdim=True) + 1e-10)
        
        if flow_mask is not None:
            feat_t_1 = feat_t_1 * flow_mask[..., None]
            flow_mask = flow_mask.view(n_batch*n_res, -1)

        if sample_multinomial:
            feat_multi_t_1 = self.sample_multinomial(
                                feat_t=feat_t_1.clamp(min=0, max=1).view(n_batch*n_res, -1),
                                flow_mask=flow_mask,
                            )
        else:
            feat_multi_t_1 = self.sample_argmax(
                                feat_t=feat_t_1.clamp(min=0, max=1).view(n_batch*n_res, -1),
                                flow_mask=flow_mask,
                            )

        feat_multi_t_1 = feat_multi_t_1.view(n_batch, n_res).long()
        return feat_t_1, feat_multi_t_1
        
        
    def forward_marginal(
        self,
        rigids_1: ru.Rigid,
        t: float,
        flow_mask: np.ndarray = None,
        as_tensor_7: bool = True,
        rigids_0: Union[ru.Rigid, None] = None,
        center_of_mass=None,
    ):
        trans_1, rot_1 = extract_trans_rots_mat(rigids_1)

        if rigids_0 is not None:
            trans_0, rot_0 = extract_trans_rots_mat(rigids_0)
        else:
            rot_0 = None
            trans_0 = None


        rot_t = self._so3_fm.forward_marginal(
            rot_1, t, rot_0=rot_0,
        )
        rot_vectorfield_scaling = self._so3_fm.vectorfield_scaling(t)
            

        trans_t = self._r3_fm.forward_marginal(
            trans_1, t, x_0=trans_0, center_of_mass=center_of_mass,
        )
        trans_vectorfield_scaling = self._r3_fm.vectorfield_scaling(t)

        if flow_mask is not None:
            rot_t = self._apply_mask(rot_t, rot_1, flow_mask[..., None])
            trans_t = self._apply_mask(trans_t, trans_1, flow_mask[..., None])

        rigids_t = ru.Rigid(
            rots=ru.Rotation(rot_mats=rot_t),
            trans=trans_t,
        )
        if as_tensor_7:
            rigids_t = rigids_t.to_tensor_7()

        return {
            "rigids_t": rigids_t,
            "rot_t": rot_t,
            "trans_t": trans_t,
            "trans_vectorfield_scaling": trans_vectorfield_scaling,
            "rot_vectorfield_scaling": rot_vectorfield_scaling,
        }

    def _apply_mask(self, x_diff, x_fixed, diff_mask):
        return diff_mask * x_diff + (1 - diff_mask) * x_fixed

    def vectorfield_scaling(self, t):
        rot_vectorfield_scaling = self._so3_fm.vectorfield_scaling(t)
        trans_vectorfield_scaling = self._r3_fm.vectorfield_scaling(t)
        return rot_vectorfield_scaling, trans_vectorfield_scaling

    def calc_trans_vectorfield(self, trans_t, trans_1, t, use_torch=False, scale=True):
        return self._r3_fm.vectorfield(
            trans_t, trans_1, t, use_torch=use_torch, scale=scale
        )

    def calc_rot_vectorfield(self, rot_t, rot_1, t):
        return self._so3_fm.vectorfield(rot_t, rot_1, t)

    def reverse_euler(
        self,
        rigid_t: ru.Rigid,
        rot: np.ndarray,
        trans: np.ndarray,
        t: float,
        dt: float,
        flow_mask: np.ndarray = None,
        center: bool = True,
        noise_scale: float = 1.0,
        context: torch.Tensor = None,
        center_of_mass = None,
    ):
        trans_t, rot_t = extract_trans_rots_mat(rigid_t)
        trans_t_1 = self._r3_fm.reverse_euler(
            x_1=trans,
            x_t=trans_t,
            t=t,
            dt=dt,
            center=center,
            noise_scale=noise_scale,
            center_of_mass=center_of_mass,
        )

        rot_t_1 = self._so3_fm.reverse_euler(
            rot_1=rot,
            rot_t=rot_t,
            t=t,
            dt=dt,
            noise_scale=noise_scale,
        )

        if flow_mask is not None:
            trans_t_1 = self._apply_mask(trans_t_1, trans_t, flow_mask[..., None])
            rot_t_1 = self._apply_mask(rot_t_1, rot_t, flow_mask[..., None, None])
        return (rot_t_1, trans_t_1, assemble_rigid_mat(rot_t_1, trans_t_1))
        

    def reverse_euler(
        self,
        rigid_t: ru.Rigid,
        rot: np.ndarray,
        trans: np.ndarray,
        t: float,
        dt: float,
        flow_mask: np.ndarray = None,
        center: bool = True,
        noise_scale: float = 1.0,
        context: torch.Tensor = None,
        center_of_mass = None,
        rot_sample_schedule = 'exp',
        trans_sample_schedule = 'linear',
    ):
        trans_t, rot_t = extract_trans_rots_mat(rigid_t)
        trans_t_1 = self._r3_fm.reverse_euler(
            x_1=trans,
            x_t=trans_t,
            t=t,
            dt=dt,
            center=center,
            noise_scale=noise_scale,
            center_of_mass=center_of_mass,
            sample_schedule=trans_sample_schedule,
        )

        rot_t_1 = self._so3_fm.reverse_euler(
            rot_1=rot,
            rot_t=rot_t,
            t=t,
            dt=dt,
            noise_scale=noise_scale,
            sample_schedule=rot_sample_schedule,
        )

        if flow_mask is not None:
            trans_t_1 = self._apply_mask(trans_t_1, trans_t, flow_mask[..., None])
            rot_t_1 = self._apply_mask(rot_t_1, rot_t, flow_mask[..., None, None])
        return (rot_t_1, trans_t_1, assemble_rigid_mat(rot_t_1, trans_t_1))
    

    def reverse(
        self,
        rigid_t: ru.Rigid,
        rot_vectorfield: np.ndarray,
        trans_vectorfield: np.ndarray,
        t: float,
        dt: float,
        flow_mask: np.ndarray = None,
        center: bool = True,
        noise_scale: float = 1.0,
        context: torch.Tensor = None,
        center_of_mass = None,
    ):
        trans_t, rot_t = extract_trans_rots_mat(rigid_t)

        rot_t_1 = self._so3_fm.reverse(
            rot_t=rot_t,
            v_t=rot_vectorfield,
            t=t,
            dt=dt,
            noise_scale=noise_scale,
        )

        trans_t_1 = self._r3_fm.reverse(
            x_t=trans_t,
            v_t=trans_vectorfield,
            t=t,
            dt=dt,
            center=center,
            noise_scale=noise_scale,
            center_of_mass=center_of_mass,
        )

        if flow_mask is not None:
            trans_t_1 = self._apply_mask(trans_t_1, trans_t, flow_mask[..., None])
            rot_t_1 = self._apply_mask(rot_t_1, rot_t, flow_mask[..., None, None])
        return (rot_t_1, trans_t_1, assemble_rigid_mat(rot_t_1, trans_t_1))

    
    def sample_ref(
        self,
        n_samples: int,
        # impute: ru.Rigid = None,
        flow_mask: np.ndarray = None,
        as_tensor_7: bool = False,
        center_of_mass=None,
    ):

        rot_ref = self._so3_fm.sample_ref(n_samples=n_samples)
        trans_ref = self._r3_fm.sample_ref(n_samples=n_samples)


        if center_of_mass is not None:
            trans_ref = trans_ref - center_of_mass

        if flow_mask is not None:
            rot_ref = self._apply_mask(rot_ref, rot_impute, flow_mask[..., None])
            trans_ref = self._apply_mask(trans_ref, trans_impute, flow_mask[..., None])
        trans_ref = self._r3_fm._unscale(trans_ref)
        rigids_t = assemble_rigid_mat(rot_ref, trans_ref)
        if as_tensor_7:
            rigids_t = rigids_t.to_tensor_7()
            
        return {
            "rigids_t": rigids_t,
            "rot_t": rot_ref,
            "trans_t": trans_ref,
        }
