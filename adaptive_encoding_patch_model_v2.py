# adaptive_encoding_patch_model1.py
import torch
import torch.nn as nn
from torch.nn import functional as F


# -------------------------
# 1) Sobel 기반 Boundary (기존)
# -------------------------
class BoundaryScoreModule(nn.Module):
    """
    입력 RGB에서 Sobel gradient 기반으로 boundary score 계산.
    - 8x8 avg pool -> 2x2 max pool => 최종 16x16 patch grid와 동일한 해상도.
    - per-image 정규화 후 threshold로 boundary mask 생성.
    """

    def __init__(self, sub_patch: int = 8, thresh: float = 0.0):
        super().__init__()
        self.sub_patch = sub_patch
        self.thresh = thresh

        sobel_x = torch.tensor(
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]],
            dtype=torch.float32,
        )
        sobel_y = torch.tensor(
            [[-1, -2, -1],
             [ 0,  0,  0],
             [ 1,  2,  1]],
            dtype=torch.float32,
        )

        self.register_buffer("kx", sobel_x.view(1, 1, 3, 3), persistent=False)
        self.register_buffer("ky", sobel_y.view(1, 1, 3, 3), persistent=False)

    def forward(self, x: torch.Tensor, patch_size: int = 16):
        """
        x: [B,3,H,W] (입력 RGB)
        return:
          boundary_mask [B,1,H/patch_size,W/patch_size]
          score_16, score_8, grad_mag
        """
        B, C, H, W = x.shape

        # grayscale
        if C == 3:
            r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
            gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        else:
            gray = x[:, :1]

        gx = F.conv2d(gray, self.kx, padding=1)
        gy = F.conv2d(gray, self.ky, padding=1)
        grad_mag = torch.sqrt(gx * gx + gy * gy + 1e-6)  # [B,1,H,W]

        sp = self.sub_patch
        score_8 = F.avg_pool2d(grad_mag, kernel_size=sp, stride=sp)          # [B,1,H/8,W/8]
        score_16 = F.max_pool2d(score_8, kernel_size=2, stride=2)            # [B,1,H/16,W/16]

        flat = score_16.view(B, -1)
        mean = flat.mean(dim=1, keepdim=True).view(B, 1, 1, 1)
        std = flat.std(dim=1, keepdim=True).view(B, 1, 1, 1)
        score_norm = (score_16 - mean) / (std + 1e-6)

        boundary_mask = (score_norm > self.thresh).float()
        return boundary_mask, score_16, score_8, grad_mag


# -------------------------
# 2) Canny-like (GPU-friendly) Boundary
# -------------------------
def _make_gaussian_kernel(ksize: int, sigma: float, device=None, dtype=torch.float32):
    assert ksize % 2 == 1
    ax = torch.arange(ksize, device=device, dtype=dtype) - (ksize // 2)
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum().clamp_min(1e-12)
    return kernel


class CannyApproxBoundaryScoreModule(nn.Module):
    """
    Torch-only / GPU-friendly Canny-style boundary mask.
    - Gaussian blur (fixed)
    - Sobel grad
    - (soft) non-max suppression 근사
    - (soft) double threshold + hysteresis 근사
    - sub_patch pooling -> 16x16 grid
    """

    def __init__(
        self,
        sub_patch: int = 8,
        gaussian_ksize: int = 5,
        gaussian_sigma: float = 1.0,
        low_thr: float = 0.10,
        high_thr: float = 0.30,
        thr_sharpness: float = 20.0,
        hyst_iters: int = 2,
        zscore_thresh: float = 0.0,
    ):
        super().__init__()
        self.sub_patch = sub_patch
        self.low_thr = low_thr
        self.high_thr = high_thr
        self.thr_sharpness = thr_sharpness
        self.hyst_iters = hyst_iters
        self.zscore_thresh = zscore_thresh

        sobel_x = torch.tensor([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1],
                                [ 0,  0,  0],
                                [ 1,  2,  1]], dtype=torch.float32)
        self.register_buffer("kx", sobel_x.view(1, 1, 3, 3), persistent=False)
        self.register_buffer("ky", sobel_y.view(1, 1, 3, 3), persistent=False)

        g = _make_gaussian_kernel(gaussian_ksize, gaussian_sigma)
        self.register_buffer("gk", g.view(1, 1, gaussian_ksize, gaussian_ksize), persistent=False)

    def forward(self, x: torch.Tensor, patch_size: int = 16):
        """
        x: [B,3,H,W]
        return:
          boundary_mask [B,1,H/patch_size,W/patch_size]
          score_16, score_8, grad_mag
        """
        B, C, H, W = x.shape

        # grayscale
        if C == 3:
            r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
            gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        else:
            gray = x[:, :1]

        # gaussian blur
        pad = self.gk.shape[-1] // 2
        gray_blur = F.conv2d(gray, self.gk, padding=pad)

        # sobel grad
        gx = F.conv2d(gray_blur, self.kx, padding=1)
        gy = F.conv2d(gray_blur, self.ky, padding=1)
        grad_mag = torch.sqrt(gx * gx + gy * gy + 1e-6)  # [B,1,H,W]

        # per-image min-max normalize to ~[0,1]
        flat = grad_mag.view(B, -1)
        gmin = flat.min(dim=1, keepdim=True).values.view(B, 1, 1, 1)
        gmax = flat.max(dim=1, keepdim=True).values.view(B, 1, 1, 1)
        grad_n = (grad_mag - gmin) / (gmax - gmin + 1e-6)

        # soft NMS approximation
        local_max = F.max_pool2d(grad_n, kernel_size=3, stride=1, padding=1)
        nms_soft = torch.sigmoid(self.thr_sharpness * (grad_n - local_max + 1e-3))
        edge_strength = grad_n * nms_soft

        # soft double threshold
        strong = torch.sigmoid(self.thr_sharpness * (edge_strength - self.high_thr))
        weak = torch.sigmoid(self.thr_sharpness * (edge_strength - self.low_thr))

        # hysteresis approximation via dilation
        cur = strong
        for _ in range(self.hyst_iters):
            dil = F.max_pool2d(cur, kernel_size=3, stride=1, padding=1)
            cur = torch.clamp(dil * weak, 0.0, 1.0)
        canny_like = cur  # [B,1,H,W]

        # pooling to patch grid
        sp = self.sub_patch
        score_8 = F.avg_pool2d(canny_like, kernel_size=sp, stride=sp)     # [B,1,H/8,W/8]
        score_16 = F.max_pool2d(score_8, kernel_size=2, stride=2)         # [B,1,H/16,W/16]

        # z-score threshold for final binary mask
        flat16 = score_16.view(B, -1)
        mean = flat16.mean(dim=1, keepdim=True).view(B, 1, 1, 1)
        std = flat16.std(dim=1, keepdim=True).view(B, 1, 1, 1)
        score_norm = (score_16 - mean) / (std + 1e-6)

        boundary_mask = (score_norm > self.zscore_thresh).float()
        return boundary_mask, score_16, score_8, grad_mag


# -------------------------
# 3) Adaptive Patch Encoder (ROD)
# -------------------------
class RODSegAdaptivePatch(nn.Module):
    """
    feature=(x,out) 형태를 입력으로 받아, boundary 위치만 local refine으로 강화.
    """

    def __init__(
        self,
        model_type: str = "vits",
        boundary_thresh: float = 0.0,
        edge_mode: str = "canny",  # "canny" or "sobel"
        # canny params
        low_thr: float = 0.10,
        high_thr: float = 0.30,
        thr_sharpness: float = 20.0,
        hyst_iters: int = 2,
    ):
        super().__init__()

        embed_size = {
            "vitt": 256, "vits": 256,
            "vit_b": 256, "vit_l": 256, "vit_h": 256
        }
        if model_type not in embed_size:
            raise ValueError(f"Unknown model_type: {model_type}")
        embed_dim = embed_size[model_type]
        print(embed_dim)

        # boundary module 선택
        if edge_mode == "sobel":
            self.boundary_module = BoundaryScoreModule(sub_patch=8, thresh=boundary_thresh)
        elif edge_mode == "canny":
            self.boundary_module = CannyApproxBoundaryScoreModule(
                sub_patch=8,
                low_thr=low_thr,
                high_thr=high_thr,
                thr_sharpness=thr_sharpness,
                hyst_iters=hyst_iters,
                zscore_thresh=boundary_thresh,
            )
        else:
            raise ValueError(f"Unknown edge_mode: {edge_mode}")

        # local refine block
        self.local_refine = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

        self.patch_size = 16

    def forward(self, image: torch.Tensor, feature: torch.Tensor):
        """
        image: [B,3,H,W] (SAM encoder 입력과 같은 이미지 텐서)
        feature: (x,out) where x is [B,C,h,w]
        return: (feats_refined, out)
        """
        x, out = feature
        B, C, h, w = x.shape

        boundary_mask, _16, _8, _grad = self.boundary_module(image, patch_size=self.patch_size)

        # feature 해상도에 맞추기
        if boundary_mask.shape[-2:] != (h, w):
            boundary_mask = F.interpolate(boundary_mask, size=(h, w), mode="nearest")

        # 채널 broadcast
        if boundary_mask.shape[1] == 1:
            boundary_mask = boundary_mask.expand(-1, C, -1, -1)

        local_res = self.local_refine(x)
        feats_refined = x + local_res * boundary_mask
        return feats_refined, out

    def boundary_analysis(self, image: torch.Tensor, feature: torch.Tensor):
        x, out = feature
        boundary_mask, s16, s8, grad = self.boundary_module(image, patch_size=self.patch_size)
        return boundary_mask, s16, s8, grad
        # ---- DEBUG: 어떤 모듈이 선택됐는지 확인 ----
        print(
            f"[AdaptivePatch] edge_mode={edge_mode} | "
            f"boundary_module={self.boundary_module.__class__.__name__} | "
            f"boundary_thresh={boundary_thresh} | "
            f"low/high=({low_thr},{high_thr}) | "
            f"hyst_iters={hyst_iters}"
        )
