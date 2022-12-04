FROM rocm/pytorch:latest

# build triton
RUN export TRITON_USE_ROCM=ON

# Unit Tests 
# to run unit tests
# 1. build this Dockerfile
#    docker build --build-arg -f triton_rocm_20-52.Dockerfile -t triton_rocm52 .
# 2. run docker container
#    docker run -it --rm --network=host --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --name triton --ipc=host --shm-size 16G --device=/dev/kfd --device=/dev/dri triton_rocm52:latest
# 3. run core unit tests on a rocm machine
#    cd ~/triton/python
#    a. To execute all test cases
#        pytest --verbose test/unit/language/test_core.py | tee test_core.log
#    b. To execute individual cases
#        pytest --verbose test/unit/language/test_core.py::test_empty_kernel[int8]
