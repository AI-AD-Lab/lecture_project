# 기본 이미지: NVIDIA CUDA + cuDNN + Ubuntu 기반
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# 기본 설정
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_ROOT_USER_ACTION=ignore

# 기본 패키지 업데이트 및 Python 설치
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv python3-dev git wget curl \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*


# pip 업그레이드
RUN pip3 install --upgrade pip setuptools wheel

# PyTorch 및 TorchVision 설치 (CUDA 12.1용)
# CUDA 버전에 맞는 최신 PyTorch를 설치합니다.
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# SAM (Segment Anything Model) 종속성 설치
RUN git clone https://github.com/facebookresearch/segment-anything.git /workspace/segment-anything
WORKDIR /workspace/segment-anything

# SAM requirements 설치
RUN pip install -e .
RUN pip install opencv-python pycocotools matplotlib onnxruntime onnx einops

# 기본 Python 환경 확인용
RUN python3 -c "import torch, torchvision; print('PyTorch:', torch.__version__, 'TorchVision:', torchvision.__version__, 'CUDA available:', torch.cuda.is_available())"

# 작업 디렉토리 설정
WORKDIR /workspace

# 기본 명령어
CMD ["/bin/bash"]
