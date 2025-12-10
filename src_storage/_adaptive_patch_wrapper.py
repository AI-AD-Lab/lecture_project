# 1단계: 기존 EfficientSAM 모델 로드 (그대로)
base_sam = build_efficient_sam(
    encoder_patch_embed_dim=384,
    encoder_num_heads=6,
    checkpoint="efficient_sam_vits.pth",  # 기존 가중치
)

# 2단계: 새 래퍼 모델 정의
class AdaptiveEfficientSam(nn.Module):
    def __init__(self, base_sam: EfficientSam, ap_module: nn.Module):
        super().__init__()
        self.base_sam = base_sam          # 기존 효율 SAM 전체
        self.ap_module = ap_module        # gVIT/Adaptive Patch 기반 모듈 (중요 패치 점수 등)

    def get_image_embeddings(self, batched_images):
        # 기존 preprocess + image_encoder 그대로 사용
        embeddings = self.base_sam.get_image_embeddings(batched_images)
        # embeddings: [B, C, H', W']

        # 여기에 adaptive patch 로직 적용 (예: 중요 패치 강조)
        # ap_mask: [B,1,H',W']
        ap_mask = self.ap_module(embeddings, batched_images)
        # 중요 위치만 강화
        embeddings_refined = embeddings + embeddings * ap_mask

        return embeddings_refined

    def forward(self, batched_images, batched_points, batched_point_labels,
                scale_to_original_image_size: bool = True):
        # 1) AP를 적용한 embedding 계산
        image_embeddings = self.get_image_embeddings(batched_images)

        # 2) 나머지 SAM decoder 로직은 기존 코드 재활용
        batch_size, _, input_h, input_w = batched_images.shape
        return self.base_sam.predict_masks(
            image_embeddings,
            batched_points,
            batched_point_labels,
            multimask_output=True,
            input_h=input_h,
            input_w=input_w,
            output_h=input_h if scale_to_original_image_size else -1,
            output_w=input_w if scale_to_original_image_size else -1,
        )