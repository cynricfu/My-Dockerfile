FROM nvcr.io/nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ARG MINIFORGE_ARCH=x86_64
ARG MINIFORGE_VERSION=25.3.0-3
ARG PYTHON_VERSION=3.11.13
ARG PIP_VERSION=25.1.1

ARG TRANSFORMERS_VERSION=4.52.4
ARG ACCELERATE_VERSION=1.7.0
ARG DATASETS_VERSION=3.6.0
ARG EVALUATE_VERSION=0.4.3
ARG SCIKIT_LEARN_VERSION=1.7.0
ARG DEEPSPEED_VERSION=0.17.1
ARG FLASH_ATTN_VERSION=2.7.4.post1
ARG CUTLASS_VERSION=3.9.2

# Install necessary apt packages
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    git-lfs \
    libaio-dev \
    curl \
    libibverbs-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
# Setup environment for CUTLASS
ENV CUTLASS_PATH=/opt/cutlass
RUN mkdir -pv ${CUTLASS_PATH}
# Setup environment for conda
ENV CONDA_DIR /opt/conda
ENV PATH "${CONDA_DIR}/bin:${PATH}"
RUN mkdir -pv ${CONDA_DIR} \
    && echo ". /opt/conda/etc/profile.d/conda.sh" >> ${HOME}/.bashrc \
    && echo ". /opt/conda/etc/profile.d/conda.sh" >> /etc/profile \
    && echo "conda activate base" >> ${HOME}/.bashrc \
    && echo "conda activate base" >> /etc/profile

# Install CUTLASS
RUN git clone --branch v${CUTLASS_VERSION} https://github.com/NVIDIA/cutlass.git /opt/cutlass
# Install Conda
RUN curl -fsSL "https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-${MINIFORGE_VERSION}-Linux-${MINIFORGE_ARCH}.sh" -o /tmp/Miniforge3.sh \
    && curl -fsSL "https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-${MINIFORGE_VERSION}-Linux-${MINIFORGE_ARCH}.sh.sha256" -o /tmp/Miniforge3.sh.sha256 \
    && echo "$(cat /tmp/Miniforge3.sh.sha256 | awk '{ print $1; }')  /tmp/Miniforge3.sh" | sha256sum -c - \
    && rm /tmp/Miniforge3.sh.sha256 \
    && /bin/bash /tmp/Miniforge3.sh -b -f -p ${CONDA_DIR} \
    && rm /tmp/Miniforge3.sh \
    && conda config --system --set auto_update_conda false \
    && conda config --system --set show_channel_urls true \
    && echo "python ==${PYTHON_VERSION}" >> ${CONDA_DIR}/conda-meta/pinned \
    && conda install -y -q python=${PYTHON_VERSION} pip=${PIP_VERSION} \
    && conda update -y -q --all \
    && conda clean -a -f -y
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

COPY --chown=root:root nvidia_entrypoint.sh /opt/nvidia/
RUN chmod 755 /opt/nvidia/nvidia_entrypoint.sh
RUN echo 'umask 0000' >> /etc/profile && \
    echo 'umask 0000' >> /root/.bashrc

ENV HF_HOME=/data/hf_home
