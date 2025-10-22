# nerfstudio/utils/clarity_map.py
"""
ClarityTracker for Gaussian Splatting (splatfacto) integration.

Features:
 - Robust extraction of gaussian parameter tensors from a variety of splatfacto field names.
 - Hutchinson estimator for J^T J (per-gaussian block) using autodiff on a low-res render.
 - Block-diagonal Laplace-style covariance update: H^+ = H^- + (1/sigma2) * J^T J
 - Per-gaussian clarity q_k = 1 / (1 + |P_k|^{1/d_k})
 - Non-blocking Open3D pointcloud viewer to visualize clarity as color per Gaussian.
 - Saves per-step artifacts (P_blocks, qk, pointcloud .ply) into outdir.
"""

import os
import logging
from typing import Optional, Dict, Tuple, Any

import numpy as np
import torch
from torch import Tensor
from torch.linalg import inv, slogdet

# open3d for visualization
try:
    import open3d as o3d  # type: ignore
    _OPEN3D_AVAILABLE = True
except Exception:
    o3d: Any = None
    _OPEN3D_AVAILABLE = False

logger = logging.getLogger("clarity_tracker")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(ch)


DK = 59  # ClarSplat parameter block size per Gaussian (position, scale, quat, alpha, SH etc.)
DEFAULT_SIGMA2 = 1e-4


class ClarityTracker:
    def __init__(
        self,
        model,
        outdir: str = "outputs/clarity",
        voxel_size: float = 0.05,
        device: Optional[torch.device] = None,
        update_every_n_steps: int = 1,
        hutchinson_samples: int = 2,
        low_res_render: Tuple[int, int] = (160, 120),
        sigma2: float = DEFAULT_SIGMA2,
    ):
        """
        model: the splatfacto model instance (the tracker will try to extract parameters from it)
        outdir: where to write per-step artifacts
        voxel_size: for optional rasterization (unused by default)
        update_every_n_steps: apply update every N global steps
        hutchinson_samples: number of Hutchinson draws to estimate J^T J per view (low=fast, high=accurate)
        low_res_render: resolution tuple (W,H) to render for J estimation to save time
        """
        self.model = model
        self.outdir = outdir
        os.makedirs(self.outdir, exist_ok=True)
        self.voxel_size = voxel_size
        self.hutchinson_samples = int(hutchinson_samples)
        self.low_res_render = tuple(low_res_render)
        self.update_every_n_steps = int(update_every_n_steps)
        self.sigma2 = float(sigma2)
        self.device = device or (next(model.parameters()).device if any(True for _ in model.parameters()) else torch.device("cpu"))
        self.P_blocks: Optional[Tensor] = None  # (N, DK, DK)
        self.initialized = False
        self.step_counter = 0

        # Open3D viewer state
        self._vis = None
        self._pcd = None
        self._first_vis = True

        if not _OPEN3D_AVAILABLE:
            logger.warning("open3d not available: clarity pointcloud viewer will be disabled. Install open3d to visualize.")

        logger.info(f"ClarityTracker initialized: outdir={self.outdir}, low_res={self.low_res_render}, device={self.device}")

    # ----------------------
    # Parameter extraction
    # ----------------------
    def extract_gaussian_params(self) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Try to extract gaussian parameters from the model. Returns:
         - theta_k: Tensor (N, DK) flattened parameter vectors (best-effort)
         - meta dict with keys: 'mu' (N,3), 'scales' (N,3), 'quat' (N,4) if available
        The function will try a few common attribute names used in splat implementations.
        """
        m = self.model
        # the Splatfacto model you pasted uses self.gauss_params[...] (means, scales, quats, features_dc, features_rest, opacities)
        # Try that directly first:
        mu = None
        scales = None
        quat = None
        alpha = None
        color = None

        if hasattr(m, "gauss_params"):
            gp = m.gauss_params
            # common keys in your file:
            mu = gp.get("means", None)
            scales = gp.get("scales", None)
            quat = gp.get("quats", None)
            color = gp.get("features_dc", None)
            # features_rest are other sh coeffs but we only pull features_dc for simplicity
            alpha = gp.get("opacities", None)

        # fallback searches if above didn't work
        if mu is None:
            for name in ("means", "positions", "xyz", "centers", "mu", "pos"):
                if hasattr(m, name):
                    mu = getattr(m, name)
                    break

        # convert to tensors & device
        def _to_tensor(x):
            if x is None:
                return None
            if isinstance(x, np.ndarray):
                return torch.from_numpy(x).to(self.device)
            if isinstance(x, Tensor):
                return x.to(self.device)
            try:
                return torch.tensor(x, device=self.device)
            except Exception:
                return None

        mu = _to_tensor(mu)
        scales = _to_tensor(scales)
        quat = _to_tensor(quat)
        alpha = _to_tensor(alpha)
        color = _to_tensor(color)

        if mu is None:
            logger.error("Failed to find gaussian positions (mu). Please inspect model and adapt extractor.")
            raise RuntimeError("ClarityTracker: no gaussian positions found.")
        N = mu.shape[0]

        # Build theta_k of length DK (59). We'll place available fields into the vector; missing fields stay zero.
        theta = torch.zeros((N, DK), device=self.device)
        # layout assumption (best-effort):
        # [0:3]=mu, [3:6]=scales, [6:10]=quat (4), [10:11]=alpha (1), [11:59]=color(48)
        theta[:, 0:3] = mu if mu.shape[1] >= 3 else mu.repeat(1, 3 // mu.shape[1] + 1)[:, :3]
        if scales is not None:
            if scales.shape[1] >= 3:
                theta[:, 3:6] = scales[:, :3]
            else:
                theta[:, 3:6] = scales.repeat(1, 3 // scales.shape[1] + 1)[:, :3]
        if quat is not None:
            qlen = min(4, quat.shape[1])
            theta[:, 6:6 + qlen] = quat[:, :qlen]
        if alpha is not None:
            theta[:, 10:11] = alpha.reshape(-1, 1) if alpha.ndim == 1 else alpha[:, :1]
        if color is not None:
            clen = min(48, color.shape[1])
            theta[:, 11:11 + clen] = color[:, :clen]

        meta = {"mu": mu, "scales": scales, "quat": quat, "alpha": alpha, "color": color}
        return theta, meta

    # ----------------------
    # Initialization
    # ----------------------
    def init_covariances(self, N: int, dk: int = DK, init_scale: float = 1e6):
        """Initialize block diagonal covariance P_k = init_scale * I for each gaussian."""
        logger.info(f"Initializing P_blocks: N={N}, dk={dk}, init_scale={init_scale}")
        device = self.device
        self.P_blocks = torch.stack([torch.eye(dk, device=device) * float(init_scale) for _ in range(N)], dim=0)
        self.initialized = True

    # ----------------------
    # J^T J estimation (Hutchinson / fallback)
    # ----------------------
    def _render_lowres(self, camera):
        """
        Try to call a low-res render on the model. Many splatfacto renderers accept a 'resolution' or 'width,height' arg.
        We'll attempt several call patterns gracefully.
        Returns a tensor (H,W,3) float on self.device
        """
        m = self.model
        # Try common call signatures using model.get_outputs or model.render-like functions.
        W, H = self.low_res_render
        # Splatfacto provides get_outputs(camera) that returns dict with "rgb"
        try:
            if camera is None:
                raise RuntimeError("No camera provided for low-res render.")
            outs = m.get_outputs(camera)  # expects a Cameras object or same interface
            if isinstance(outs, dict) and "rgb" in outs:
                img = outs["rgb"]
                # img is [H, W, 3] on model.device
                if img.shape[0] != H or img.shape[1] != W:
                    # downsample
                    img_t = img.permute(2, 0, 1)[None, ...]  # [1, C, H, W]
                    import torch.nn.functional as F

                    img_ds = F.interpolate(img_t, size=(H, W), mode="bilinear", align_corners=False).squeeze(0).permute(1, 2, 0)
                    return img_ds.to(self.device)
                return img.to(self.device)
        except Exception:
            # fallback to try other generic render calls
            if hasattr(m, "render"):
                try:
                    out = m.render(camera, width=W, height=H)
                    return out.to(self.device)
                except Exception:
                    pass
                try:
                    out = m.render(camera, resolution=(W, H))
                    return out.to(self.device)
                except Exception:
                    pass
                try:
                    out = m.render(camera)
                    if isinstance(out, torch.Tensor):
                        if out.dim() == 3 and (out.shape[0] != H or out.shape[1] != W):
                            out = torch.nn.functional.interpolate(
                                out.permute(2, 0, 1).unsqueeze(0),
                                size=(H, W),
                                mode="bilinear",
                                align_corners=False,
                            ).squeeze(0).permute(1, 2, 0)
                        return out.to(self.device)
                except Exception:
                    pass
        raise RuntimeError("ClarityTracker: low-res render with model failed for tried signatures.")

    def compute_jtj_hutchinson(self, camera, pixel_sample_limit=8000, samples: Optional[int] = None) -> Tensor:
        """
        Estimate per-gaussian J^T J blocks for the current view using Hutchinson.
        Returns: JtJ_blocks: (N, DK, DK)
        This is a heuristic estimator that tries to use autograd where possible; it contains robust fallbacks.
        """
        if samples is None:
            samples = self.hutchinson_samples
        theta, meta = self.extract_gaussian_params()
        N = theta.shape[0]
        dk = theta.shape[1]

        # render and sample pixels
        try:
            img = self._render_lowres(camera)  # (H,W,3)
        except Exception as e:
            logger.warning(f"Low-res render failed: {e}. Falling back to tiny isotropic JtJ.")
            return torch.stack([torch.eye(dk, device=self.device) * 1e-6 for _ in range(N)], dim=0)

        if not torch.is_tensor(img):
            img = torch.tensor(img, device=self.device)
        pixels_flat = img.reshape(-1)  # length = M*3
        total_len = pixels_flat.shape[0]
        sample_size = min(pixel_sample_limit, total_len)
        idx = torch.randperm(total_len, device=self.device)[:sample_size]
        pixels_sampled = pixels_flat[idx]  # length sample_size

        # Storage: per-gaussian JtJ accumulator
        JtJ_blocks = torch.zeros((N, dk, dk), device=self.device)

        # We attempt an autograd-based gradient gather. This relies on model parameters having grads set by calling
        # backward on the Hutchinson probe. If param layout doesn't map cleanly to theta vector, fallback to small isotropic proxy.
        for s in range(samples):
            v = torch.randint(0, 2, (sample_size,), device=self.device, dtype=torch.float32) * 2.0 - 1.0
            s_val = torch.dot(v, pixels_sampled)
            # zero grads
            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()
            try:
                s_val.backward(retain_graph=True)
            except Exception:
                # can't backpropagate into render pixels generically
                continue

            # Try to collect grads corresponding to gaussian param tensors:
            try:
                if hasattr(self.model, "gauss_params"):
                    gp = self.model.gauss_params
                    # gather grads in the same order as populate_modules get_gaussian_param_groups
                    grads_list = []
                    for name in ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]:
                        if name in gp:
                            p = gp[name]
                            if p.grad is not None:
                                grads_list.append(p.grad.detach().reshape(p.grad.shape[0], -1))
                            else:
                                grads_list.append(torch.zeros((p.shape[0], p.numel() // p.shape[0]), device=self.device))
                    # flatten per-gaussian grads
                    # some params are multi-dim per-gaussian (e.g., features_rest has dim >1). We will concatenate per gaussian up to DK.
                    per_gauss_grads = []
                    for i in range(N):
                        parts = []
                        for block in grads_list:
                            # block shape may be (N, x) — if block rows < N, broadcast
                            if block.shape[0] == N:
                                parts.append(block[i])
                            else:
                                # fallback zeros of size inferred
                                parts.append(torch.zeros(block.shape[1], device=self.device))
                        gk = torch.cat(parts, dim=0)
                        # ensure length dk by trunc/pad
                        if gk.numel() >= dk:
                            gk = gk[:dk]
                        else:
                            pad = torch.zeros((dk - gk.numel(),), device=self.device)
                            gk = torch.cat([gk, pad], dim=0)
                        per_gauss_grads.append(gk)
                    grads_reshaped = torch.stack(per_gauss_grads, dim=0)  # (N, dk)
                    for k in range(N):
                        gk = grads_reshaped[k]
                        JtJ_blocks[k] += torch.ger(gk, gk)
                    continue  # next Hutchinson sample
            except Exception:
                # fall through to fallback
                pass

        # if JtJ_blocks is still zero, use small isotropic proxy
        if (JtJ_blocks.abs().sum() == 0).item():
            logger.warning("JtJ estimator returned zero matrix (fallback). Using small isotropic proxy.")
            for k in range(N):
                JtJ_blocks[k] = torch.eye(dk, device=self.device) * 1e-6

        return JtJ_blocks

    # ----------------------
    # Covariance update & clarity
    # ----------------------
    def update_covariances_and_clarity(self, JtJ_blocks: Tensor, reg_eps: float = 1e-6) -> Tuple[Tensor, Tensor]:
        """
        Given JtJ_blocks (N,dk,dk), update P_blocks and compute qk.
        Returns: (new_P_blocks, qk)
        """
        if self.P_blocks is None:
            N = JtJ_blocks.shape[0]
            self.init_covariances(N)

        # after init_covariances we guarantee P_blocks is not None
        assert self.P_blocks is not None
        N = self.P_blocks.shape[0]
        dk = self.P_blocks.shape[1]
        new_P = torch.empty_like(self.P_blocks)
        qk_list = torch.empty((N,), device=self.device)

        for k in range(N):
            Pk = self.P_blocks[k]
            # H_minus = inv(Pk)
            try:
                Hm = inv(Pk + reg_eps * torch.eye(dk, device=self.device))
            except Exception:
                # robust fallback: construct diagonal inverse from Pk diagonal
                diag_pk = torch.diag(Pk)
                Hm = torch.diag(1.0 / (diag_pk + reg_eps))
            Hplus = Hm + (1.0 / self.sigma2) * JtJ_blocks[k]
            # symmetrize
            Hplus = 0.5 * (Hplus + Hplus.T) + reg_eps * torch.eye(dk, device=self.device)
            # invert
            try:
                Pnew = inv(Hplus)
            except Exception as e:
                # fallback small diag
                logger.warning(f"Failed invert Hplus for gaussian {k}: {e}. Using pseudo-inverse diag fallback.")
                diag = torch.diag(torch.diag(Hplus))
                Pnew = inv(diag + reg_eps * torch.eye(dk, device=self.device))
            new_P[k] = Pnew
            # compute determinant^(1/d)
            sign, logabs = slogdet(Pnew)
            # convert to Python scalars for safe boolean checks and math
            try:
                sign_val = float(sign.item()) if isinstance(sign, torch.Tensor) else float(sign)
            except Exception:
                sign_val = float(sign)
            try:
                logabs_val = float(logabs.item()) if isinstance(logabs, torch.Tensor) else float(logabs)
            except Exception:
                logabs_val = float(logabs)

            if (sign_val <= 0) or (np.isnan(logabs_val)):
                # regularize
                Pnew = Pnew + reg_eps * torch.eye(dk, device=self.device)
                sign, logabs = slogdet(Pnew)
                try:
                    logabs_val = float(logabs.item()) if isinstance(logabs, torch.Tensor) else float(logabs)
                except Exception:
                    logabs_val = float(logabs)

            det_pow = float(np.exp(logabs_val / float(dk)))
            qk_list[k] = 1.0 / (1.0 + det_pow)

        self.P_blocks = new_P
        return new_P, qk_list

    # ----------------------
    # Visualization (Open3D)
    # ----------------------
    def _init_open3d(self):
        if not _OPEN3D_AVAILABLE:
            return
        if self._vis is None:
            self._vis = o3d.visualization.Visualizer()
            self._vis.create_window(window_name="Clarity Map (per Gaussian)", width=900, height=700, visible=True)
            self._pcd = o3d.geometry.PointCloud()
            self._vis.add_geometry(self._pcd)
            opt = self._vis.get_render_option()
            opt.point_size = 3.0
            logger.info("Open3D viewer started for clarity visualization.")

    def visualize_pointcloud(self, mu: Tensor, qk: Tensor, step: Optional[int] = None):
        """
        Visualize a colored point cloud where color encodes clarity qk (red=clear/high q, green=uncertain low q).
        Non-blocking: updates the same window when called repeatedly.
        """
        if not _OPEN3D_AVAILABLE:
            return
        self._init_open3d()
        pts = mu.detach().cpu().numpy()
        q = qk.detach().cpu().numpy()
        # colors: red = q, green = (1-q)
        colors = np.stack([q, 1.0 - q, np.zeros_like(q)], axis=1)
        if self._pcd is None or self._vis is None:
            logger.debug("Open3D objects not initialized; skipping visualization update.")
            return
        self._pcd.points = o3d.utility.Vector3dVector(pts)
        self._pcd.colors = o3d.utility.Vector3dVector(colors)
        # update geometry
        self._vis.update_geometry(self._pcd)
        self._vis.poll_events()
        self._vis.update_renderer()
        # optionally save a snapshot & pointcloud file
        if step is not None:
            fname = os.path.join(self.outdir, f"clarity_pcd_step_{step:06d}.ply")
            o3d.io.write_point_cloud(fname, self._pcd)
            logger.info(f"Saved clarity PLY: {fname}")

    # ----------------------
    # Top-level update (call from training loop)
    # ----------------------
    def step_end_callback(self, global_step: int, camera=None):
        """
        Should be called after optimizer.step() in the training loop.
        Optionally provide the camera/pose used to render the last view; otherwise the tracker will attempt
        to extract a reasonable camera from the pipeline/model (best-effort).
        """
        self.step_counter += 1
        if (global_step % self.update_every_n_steps) != 0:
            return

        # try to obtain camera object if not provided
        cam = camera
        if cam is None:
            # best-effort attempts to read from model or pipeline
            try:
                cam = getattr(self.model, "last_camera", None)
                if cam is None and hasattr(self.model, "camera"):
                    cam = getattr(self.model, "camera")
            except Exception:
                cam = None

        try:
            theta, meta = self.extract_gaussian_params()
            N = theta.shape[0]
            if not self.initialized:
                self.init_covariances(N)
            if cam is None:
                logger.debug("No camera provided to ClarityTracker.step_end_callback(); JtJ will use proxy identity update.")
                # proxy tiny JtJ so that covariances shrink slightly over time
                JtJ = torch.stack([torch.eye(DK, device=self.device) * 1e-6 for _ in range(N)], dim=0)
            else:
                # estimate JtJ using Hutchinson on the given camera
                JtJ = self.compute_jtj_hutchinson(cam, pixel_sample_limit=8000, samples=self.hutchinson_samples)
            Pnew, qk = self.update_covariances_and_clarity(JtJ)
            # visualize pointcloud
            mu = meta.get("mu", theta[:, 0:3])
            try:
                self.visualize_pointcloud(mu, qk, step=global_step)
            except Exception as e:
                logger.warning(f"Open3D visualization failed: {e}")
            # save P_blocks and qk
            try:
                if self.P_blocks is not None:
                    torch.save(self.P_blocks.detach().cpu(), os.path.join(self.outdir, f"P_blocks_step_{global_step:06d}.pt"))
                np.save(os.path.join(self.outdir, f"qk_step_{global_step:06d}.npy"), qk.detach().cpu().numpy())
            except Exception as e:
                logger.warning(f"Failed to save clarity artifacts: {e}")
        except Exception as e:
            logger.exception(f"ClarityTracker.step_end_callback failed at step {global_step}: {e}")
