FROM rocm/pytorch:latest-release

WORKDIR /workspace
USER root

# Download triton/FA_triton_benchmark_v1 and install triton
RUN git clone -b FA_triton_benchmark_v1 https://github.com/ROCmSoftwarePlatform/triton.git
RUN cd triton/python
RUN pip install -e .

# Download flash-attention/FA_triton_benchmark_v1
RUN cd /workspace
RUN git clone -b FA_triton_benchmark_v1 https://github.com/ROCmSoftwarePlatform/flash-attention.git

