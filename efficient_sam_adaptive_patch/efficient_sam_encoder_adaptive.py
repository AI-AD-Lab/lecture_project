import math
from typing import List, Type

import torch
from torch import nn, Tensor
import torch.nn.functional as F

# 기존 코드에 이미 있을 것으로 가정
from .efficient_sam_encoder import PatchEmbed, Block, LayerNorm2d, get_abs_pos


class ImageEncoderViTAdaptivePatch(nn.Module):
    """
    EfficientSAM용 ViT 인코더 + 가변 패치(중요 위치만 8x8 세분화).

    - 기본 패치 크기: patch_size (예: 16)
    - fine 패치 크기: patch_size // 2 (예: 8)
    - boundary_mask == 1 인 16x16 위치:
        -> 해당 16x16 영역을 8x8 4개로 쪼개서 임베딩한 뒤 평균/합성해서
           하나의 토큰으로 사용 (more fine detail)
    - boundary_mask == 0 인 위치:
        -> 일반 16x16 패치 임베딩 사용

    최종 토큰 개수 / grid 해상도는 모두 16x16 기준이므로
    pos_embed / decoder 구조는 기존과 동일하게 유지 가능.
    """

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_chans: int,
        patch_embed_dim: int,
        normalization_type: str,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        neck_dims: List[int],
        act_layer: Type[nn.Module],
    ) -> None:
        super().__init__()

        self.img_size = img_size
        self.base_patch_size = patch_size               # ex. 16
        self.fine_patch_size = patch_size // 2          # ex. 8
        assert self.base_patch_size % self.fine_patch_size == 0, \
            "base_patch_size는 fine_patch_size의 정수배여야 합니다 (예: 16, 8)."

        # 최종 feature grid 해상도는 16x16 기준
        self.image_embedding_size = img_size // self.base_patch_size
        self.transformer_output_dim = ([patch_embed_dim] + neck_dims)[-1]

        # -----------------------------
        # Patch Embedding (16x16, 8x8 두 가지)
        # -----------------------------
        self.patch_embed_16 = PatchEmbed(
            img_size, self.base_patch_size, in_chans, patch_embed_dim
        )
        self.patch_embed_8 = PatchEmbed(
            img_size, self.fine_patch_size, in_chans, patch_embed_dim
        )

        # -----------------------------
        # CLS 토큰 + Positional Embedding (기본은 16x16 grid 기준)
        # -----------------------------
        self.pretrain_use_cls_token = True
        self.cls_token = nn.Parameter(torch.zeros(1, 1, patch_embed_dim))

        pretrain_img_size = 224
        num_patches_16 = (pretrain_img_size // self.base_patch_size) * (
            pretrain_img_size // self.base_patch_size
        )
        num_positions = num_patches_16 + 1  # +1 for CLS
        self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, patch_embed_dim))

        # -----------------------------
        # ViT Blocks
        # -----------------------------
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(Block(patch_embed_dim, num_heads, mlp_ratio, True))

        # -----------------------------
        # Neck (기존과 동일)
        # -----------------------------
        self.neck = nn.Sequential(
            nn.Conv2d(
                patch_embed_dim,
                neck_dims[0],
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(neck_dims[0]),
            nn.Conv2d(
                neck_dims[0],
                neck_dims[0],
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(neck_dims[0]),
        )

        self.act = act_layer()

        # 파라미터 초기화 (필요하면 여기서 trunc_normal_ 등 추가)
        nn.init.zeros_(self.cls_token)
        nn.init.zeros_(self.pos_embed)

    # ------------------------------------------------------------
    # 헬퍼: [B,N,D] <-> [B,D,H,W] 변환
    # ------------------------------------------------------------
    def _tokens_to_grid(self, tokens: Tensor, grid_size: int) -> Tensor:
        """
        tokens: [B,N,D]
        return: [B,D,H,W], H=W=grid_size
        """
        B, N, D = tokens.shape
        H = W = grid_size
        assert H * W == N, f"토큰 개수 {N}와 grid {H}x{W}가 맞지 않습니다."
        x = tokens.view(B, H, W, D).permute(0, 3, 1, 2)  # [B,D,H,W]
        return x

    def _grid_to_tokens(self, x: Tensor) -> Tensor:
        """
        x: [B,D,H,W]
        return: [B,N,D]
        """
        B, D, H, W = x.shape
        tokens = x.permute(0, 2, 3, 1).reshape(B, H * W, D)
        return tokens

    # ------------------------------------------------------------
    # 8x8 토큰 4개를 합쳐 16x16 토큰 하나로 만드는 부분
    # ------------------------------------------------------------
    def merge_8_to_16_tokens(self, tokens8: Tensor) -> Tensor:
        """
        tokens8: [B, N8, D], grid8 = H/8
        return: tokens16_from8: [B, N16, D], grid16 = H/16
        """
        B, N8, D = tokens8.shape
        grid8 = int(math.sqrt(N8))      # ex. 128 (1024/8)
        H8 = W8 = grid8
        H16 = H8 // 2
        W16 = W8 // 2

        # [B,N8,D] -> [B,H8,W8,D]
        t8 = tokens8.view(B, H8, W8, D)

        # [B,H8,W8,D] -> [B,H16,2,W16,2,D]
        t8 = t8.view(B, H16, 2, W16, 2, D)

        # 2x2 블록 평균 → [B,H16,W16,D]
        t8_merged = t8.mean(dim=(2, 4))

        # 다시 [B,N16,D] 로
        tokens16_from8 = t8_merged.view(B, H16 * W16, D)
        return tokens16_from8

    # ------------------------------------------------------------
    # boundary_mask를 이용해서 16x16 vs 8x8 기반 토큰을 위치별로 선택
    # ------------------------------------------------------------
    def adaptive_patch_tokens(
        self,
        x: Tensor,
        boundary_mask: Tensor,
    ) -> Tensor:
        """
        x: [B,3,H,W]
        boundary_mask: [B,1,H16,W16] (0/1)
        return: fused_tokens: [B,N16,D]
        """
        # 1) 16x16 패치 토큰
        tokens16 = self.patch_embed_16(x)      # [B,N16,D]
        B, N16, D = tokens16.shape
        grid16 = int(math.sqrt(N16))          # H16=W16

        # 2) 8x8 패치 토큰 → 16x16 grid로 합치기
        tokens8 = self.patch_embed_8(x)       # [B,N8,D]
        tokens16_from8 = self.merge_8_to_16_tokens(tokens8)  # [B,N16,D]

        # 3) boundary_mask: [B,1,H16,W16] → [B,N16,1]
        if boundary_mask.shape[-2:] != (grid16, grid16):
            boundary_mask_resized = F.interpolate(
                boundary_mask,
                size=(grid16, grid16),
                mode="nearest",
            )
        else:
            boundary_mask_resized = boundary_mask

        mask = (
            boundary_mask_resized.permute(0, 2, 3, 1)  # [B,H16,W16,1]
            .reshape(B, N16, 1)                        # [B,N16,1]
            .to(tokens16.device)
        )

        # 4) 위치별로 선택 (mask=1 → 8x8 기반, mask=0 → 16x16 기반)
        fused_tokens = tokens16 * (1.0 - mask) + tokens16_from8 * mask  # [B,N16,D]
        return fused_tokens

    # ------------------------------------------------------------
    # pos_embed를 현재 grid 크기에 맞게 interpolation
    # ------------------------------------------------------------
    def interpolate_pos_encoding(self, H: int, W: int) -> Tensor:
        """
        H,W: 현재 patch grid 크기 (ex. 64x64)
        return: pos_embed_resized: [1, 1+H*W, D]
        """
        pos_embed = self.pos_embed  # [1, 1+N_pre, D]
        N = pos_embed.shape[1] - 1
        D = pos_embed.shape[2]
        cls_pos = pos_embed[:, :1, :]      # [1,1,D]
        patch_pos = pos_embed[:, 1:, :]    # [1,N_pre,D]

        if N == H * W:
            return pos_embed

        size_pre = int(math.sqrt(N))
        assert size_pre * size_pre == N, "pretrain pos_embed의 patch 개수가 정사각형이 아닙니다."

        patch_pos = patch_pos.reshape(1, size_pre, size_pre, D).permute(0, 3, 1, 2)
        # [1,D,H_pre,W_pre] -> interpolate -> [1,D,H,W]
        patch_pos = F.interpolate(
            patch_pos,
            size=(H, W),
            mode="bicubic",
            align_corners=False,
        )
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, H * W, D)
        pos_embed_resized = torch.cat([cls_pos, patch_pos], dim=1)  # [1,1+H*W,D]
        return pos_embed_resized

    # ------------------------------------------------------------
    # forward
    # ------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        boundary_mask: torch.Tensor | None = None,
    ):
        """
        x: [B,3,H,W], H=W=self.img_size (예: 1024)
        boundary_mask: [B,1,H16,W16] or None
        - H16 = self.img_size // 16 (예: 64)

        return:
        x:   [B, C_out, H16, W16]    (neck을 지난 최종 feature)
        out: length=depth 리스트, 각 원소 [B,H16,W16,D] (block별 중간 feature)
        """
        B, C, H, W = x.shape
        assert (
            H == self.img_size and W == self.img_size
        ), "input image size must match self.img_size"

        # 1) 기본 16x16 patch embedding
        #    x16: [B, D, H16, W16]
        x16 = self.patch_embed_16(x)

        # 2) boundary_mask가 있으면, 중요한 위치만 8x8 기반으로 대체
        if boundary_mask is not None:
            # 2-1) 8x8 patch embedding
            #      x8: [B, D, H8, W8], H8 = 2*H16, W8 = 2*W16
            x8 = self.patch_embed_8(x)

            B_, D_, H8, W8 = x8.shape
            H16 = x16.shape[2]
            W16 = x16.shape[3]
            assert H8 == H16 * 2 and W8 == W16 * 2, \
                f"8x8 feat 크기 {H8}x{W8}가 16x16 기준 {H16}x{W16}와 맞지 않습니다."

            # 2-2) 8x8 4개(2x2)를 평균내서 16x16 하나로 합치기
            #      x8_merged: [B,D,H16,W16]
            x8_merged = x8.view(B_, D_, H16, 2, W16, 2).mean(dim=(3, 5))

            # 2-3) boundary_mask를 [B,1,H16,W16]로 맞추기
            if boundary_mask.shape[-2:] != (H16, W16):
                boundary_mask_resized = F.interpolate(
                    boundary_mask,
                    size=(H16, W16),
                    mode="nearest",
                )
            else:
                boundary_mask_resized = boundary_mask

            # 2-4) 채널 방향으로 broadcast
            #      mask: [B,1,H16,W16] → [B,D,H16,W16]
            mask = boundary_mask_resized.to(x16.device)
            if mask.shape[1] == 1 and D_ == x16.shape[1]:
                mask = mask.expand(-1, x16.shape[1], -1, -1)

            # 2-5) 위치별로 선택 (mask=1 → 8x8 기반, mask=0 → 16x16 기반)
            x_patches = x16 * (1.0 - mask) + x8_merged * mask   # [B,D,H16,W16]
        else:
            # boundary_mask 없으면 기존과 동일
            x_patches = x16  # [B,D,H16,W16]

        # 3) B C H W -> B H W C
        x = x_patches.permute(0, 2, 3, 1)  # [B,H16,W16,D]

        # 4) absolute positional embedding 추가
        x = x + get_abs_pos(
            self.pos_embed, self.pretrain_use_cls_token, [x.shape[1], x.shape[2]]
        )  # [B,H16,W16,D]

        num_patches = x.shape[1]
        assert x.shape[2] == num_patches

        # 5) B,H,W,C -> B, N, C (ViT block 입력)
        x = x.reshape(x.shape[0], num_patches * num_patches, x.shape[3])  # [B,N16,D]

        out = []
        for blk in self.blocks:
            x = blk(x)  # [B,N16,D]
            x_out = x.reshape(x.shape[0], num_patches, num_patches, x.shape[2])  # [B,H16,W16,D]
            out.append(x_out)

        # 6) 마지막 block 출력도 H,W grid로 reshape
        x = x.reshape(x.shape[0], num_patches, num_patches, x.shape[2])  # [B,H16,W16,D]

        # 7) neck 통과: B,H,W,C -> B,C,H,W
        x = self.neck(x.permute(0, 3, 1, 2))  # [B,C_out,H16,W16]

        return x, out
