# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import List, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

"""
class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""

    def __init__(
        self,
        img_size,
        patch_size,
        in_chans,
        embed_dim,
    ):
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            bias=True,
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        return x
"""

# 기존 PatchEmbed 대신 사용될 AP 전용 선형 임베딩
class AdaptivePatchEmbed(nn.Module):
    def __init__(self, patch_size: int, in_chans: int, embed_dim: int):
        super().__init__()
        # AP 파이프라인에서 패치 크기를 P로 통일했으므로, 
        # 임베딩 차원은 P * P * C 입니다.
        patch_dim = patch_size * patch_size * in_chans 
        # 선형 투영: (P*P*C) -> E (embed_dim)
        self.proj = nn.Linear(patch_dim, embed_dim)

    def forward(self, x: torch.Tensor):
        # x 형태는 (B, C, H, W)이며, H*W는 총 패치 수 S를 P*P로 나눈 것.
        # (B, C, H, W)를 (B, H*W, C)로 변환하여 Linear Layer에 넣기 위해 Flatten이 필요함.
        # 이 코드는 ViTUNet의 Embeddings 클래스와 유사하게,
        # 입력된 QDT 이미지를 Sequence 텐서로 변환하는 역할을 수행해야 합니다.
        
        # ViTUNet Embeddings와 동일한 Conv2d를 사용하여 QDT 이미지에서 토큰을 추출하는 방식으로 대체
        self.proj_conv = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            bias=True,
        )
        x = self.proj_conv(x)
        B, C, H_out, W_out = x.shape
        # (B, C, H_out, W_out) -> (B, H_out * W_out, C) 형태로 변환 (트랜스포머 입력)
        x = x.flatten(2).transpose(1, 2) 
        return x






class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        qkv_bias,
        qk_scale=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        act_layer=nn.GELU,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


@torch.jit.export
def get_abs_pos(
    abs_pos: torch.Tensor, has_cls_token: bool, hw: List[int]
) -> torch.Tensor:
    """
    Calculate absolute positional embeddings. If needed, resize embeddings and remove cls_token
        dimension for the original embeddings.
    Args:
        abs_pos (Tensor): absolute positional embeddings with (1, num_position, C).
        has_cls_token (bool): If true, has 1 embedding in abs_pos for cls token.
        hw (Tuple): size of input image tokens.

    Returns:
        Absolute positional embeddings after processing with shape (1, H, W, C)
    """
    h = hw[0]
    w = hw[1]
    if has_cls_token:
        abs_pos = abs_pos[:, 1:]
    xy_num = abs_pos.shape[1]
    size = int(math.sqrt(xy_num))
    assert size * size == xy_num

    if size != h or size != w:
        new_abs_pos = F.interpolate(
            abs_pos.reshape(1, size, size, -1).permute(0, 3, 1, 2),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        )
        return new_abs_pos.permute(0, 2, 3, 1)
    else:
        return abs_pos.reshape(1, h, w, -1)


# Image encoder for efficient SAM.
class AdaptiveImageEncoderViT(nn.Module):
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

        # AP 전용: 고정된 시퀀스 길이 S를 받습니다.
        target_seq_len: int = 576,

    ) -> None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            patch_embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            act_layer (nn.Module): Activation layer.
        """
        super().__init__()

        self.img_size = img_size
        # AP는 이미 시퀀스 길이를 S로 고정했으므로, 이미지 임베딩 크기 H_out, W_out을 S에서 역산합니다.
        self.num_patches = target_seq_len
        self.image_embedding_size = int(math.sqrt(self.num_patches))



        # self.image_embedding_size = img_size // ((patch_size if patch_size > 0 else 1))
        self.transformer_output_dim = ([patch_embed_dim] + neck_dims)[-1]
        self.pretrain_use_cls_token = True
        pretrain_img_size = 224



        # 1. PatchEmbed를 AdaptivePatchEmbed로 대체 (또는 ViTUNet의 Conv2d 방식 사용)
        self.patch_embed = nn.Conv2d(
            in_chans,
            patch_embed_dim,
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            bias=True,
        )
        
        
        # ----------------------------------------------------

        self.transformer_output_dim = ([patch_embed_dim] + neck_dims)[-1]
        self.pretrain_use_cls_token = False # AP는 보통 cls_token을 사용하지 않음 (ViTUNet 기준)
        
        # AP는 target_seq_len에 맞게 고정된 위치 임베딩을 사용합니다.
        # self.num_patches = target_seq_len (576)
        num_positions = self.num_patches
        """
        # Initialize absolute positional embedding with pretrain image size.
        num_patches = (pretrain_img_size // patch_size) * (
            pretrain_img_size // patch_size
        )
        num_positions = num_patches + 1
        """
        self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, patch_embed_dim))
        self.blocks = nn.ModuleList()
        for i in range(depth):
            vit_block = Block(patch_embed_dim, num_heads, mlp_ratio, True)
            self.blocks.append(vit_block)
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

# forward 함수는 QDT 텐서를 입력으로 받습니다.
    def forward(self, x_qdt: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # QDT 입력 형태: (B, C, P, S*P) 또는 (B, C, H_qdt, W_qdt)
        # H_qdt * W_qdt는 총 패치 수 S * P * P가 아닙니다.
        
        # 1. Patch Embed (QDT 텐서를 시퀀스로 변환)
        x = self.patch_embed(x_qdt) 
        # x 형태는 (B, E, H_out, W_out). H_out * W_out = S (시퀀스 길이)
        
        B, C_out, H_out, W_out = x.shape
        num_patches = H_out * W_out 
        
        # (B, E, H_out, W_out) -> (B, H_out * W_out, E) = (B, S, E)
        x = x.flatten(2).transpose(1, 2)

        # 2. 위치 임베딩 추가 (고정된 S=576을 사용)
        x = x + self.pos_embed
        
        # 3. 트랜스포머 블록 실행
        out_features = []
        for blk in self.blocks:
            x = blk(x)
            
            # 중간 특징 맵 저장 (ViTUNet의 z3, z6, z9, z12와 유사)
            # (B, S, E) -> (B, E, H_out, W_out)로 재구성
            x_out = x.transpose(1, 2).reshape(B, C_out, H_out, W_out)
            out_features.append(x_out) 

        # 4. Neck 및 최종 출력 (SAM 스타일)
        # 최종 x 형태는 (B, S, E)이므로 2D로 재구성
        x = x.transpose(1, 2).reshape(B, C_out, H_out, W_out) 
        final_x = self.neck(x)
        
        # AP에서는 U-Net 디코더에 중간 특징 맵(out_features)을 전달합니다.
        # final_x는 최종 인코딩된 특징 맵입니다.
        return final_x, out_features



"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert (
            x.shape[2] == self.img_size and x.shape[3] == self.img_size
        ), "input image size must match self.img_size"
        x = self.patch_embed(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        x = x + get_abs_pos(
            self.pos_embed, self.pretrain_use_cls_token, [x.shape[1], x.shape[2]]
        )                                                                           #x.torch.size([2,64,64,348])
       


        num_patches = x.shape[1]
        assert x.shape[2] == num_patches
        x = x.reshape(x.shape[0], num_patches * num_patches, x.shape[3])            #x.torch.size([2,4096,348])
        out = []                                                                    #out.torch.size([2,64,64,348])*12
        for blk in self.blocks:
            x = blk(x)
            x_out = x.reshape(x.shape[0], num_patches, num_patches, x.shape[2])
            out.append(x_out)
        x = x.reshape(x.shape[0], num_patches, num_patches, x.shape[2])             #x.torch.size([2,64,64,348])
        x = self.neck(x.permute(0, 3, 1, 2))
        return x,out
"""

