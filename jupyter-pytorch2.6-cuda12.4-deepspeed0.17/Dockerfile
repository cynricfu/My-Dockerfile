FROM ghcr.io/kubeflow/kubeflow/notebook-servers/jupyter

ARG TRANSFORMERS_VERSION=4.52.4
ARG ACCELERATE_VERSION=1.7.0
ARG DATASETS_VERSION=3.6.0
ARG EVALUATE_VERSION=0.4.3
ARG SCIKIT_LEARN_VERSION=1.7.0
ARG DEEPSPEED_VERSION=0.17.1
ARG FLASH_ATTN_VERSION=2.7.4.post1
ARG CUTLASS_VERSION=3.9.2

# nvidia container toolkit
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=12.4"

# Switch to root user to install necessary apt packages
USER root
# Install necessary apt packages
RUN apt-get update && apt-get install -y \
    sudo \
    build-essential \
    git \
    git-lfs \
    libaio-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
# Setup environment for CUTLASS
ENV CUTLASS_PATH=/opt/cutlass
RUN mkdir -pv ${CUTLASS_PATH} \
    && chown -R ${NB_USER}:${NB_GID} ${CUTLASS_PATH}
# Switch back to the jupyter user
USER $NB_UID

# Install CUDA 12.4
RUN mamba install -y -q cuda -c nvidia/label/cuda-12.4.1 \
    && mamba clean -a -f -y
# Install CUTLASS
RUN git clone --branch v${CUTLASS_VERSION} https://github.com/NVIDIA/cutlass.git /opt/cutlass
# Install Python packages
RUN python3 -m pip install --quiet --no-cache-dir \
    torch==2.6.0 \
    torchvision==0.21.0 \
    torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124
RUN python3 -m pip install --no-cache-dir \
    transformers==${TRANSFORMERS_VERSION} \
    accelerate==${ACCELERATE_VERSION} \
    datasets==${DATASETS_VERSION} \
    evaluate==${EVALUATE_VERSION} \
    scikit-learn==${SCIKIT_LEARN_VERSION} \
    deepspeed==${DEEPSPEED_VERSION}
RUN python3 -m pip install --quiet --no-cache-dir \
    flash-attn==${FLASH_ATTN_VERSION} --no-build-isolation
RUN python3 -m pip install --quiet --no-cache-dir \
    kfp==2.11.0 \
    jupyterlab-git==0.50.2

USER root
# Insert umask to the jupyterlab run
COPY --chown=${NB_USER}:${NB_GID} s6/ /etc
RUN chmod 755 /etc/services.d/jupyterlab/run
# 设置系统级 umask
RUN echo 'umask 0000' >> /etc/profile
# 设置用户 .bashrc
RUN echo 'umask 0000' >> /home/${NB_USER}/.bashrc
RUN echo 'umask 0000' >> /root/.bashrc
# 设置 jovyan 及 root 用户密码
RUN echo "jovyan:jovyan" | chpasswd
RUN echo "root:root" | chpasswd
RUN usermod -aG sudo jovyan
RUN echo 'jovyan ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER $NB_UID

ENV HF_HOME=/home/${NB_USER}/pvc-shared/hf_home
