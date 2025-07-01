#!/bin/sh
# Shell script which is run when an env is activated or deactivated. This is used to set env vars which let
# python know where the cuda libraries are. This is primarily needed for libraries (like ammo) which need to
# compile cuda kernals at runtime.
#
# We add this to /THE_CONDA_ENV/etc/conda/activate.d/activate_env.sh so it automatically runs when the env
# is activated

# Save the old env vars so we can revert to them on deactivation
export OLD_LD_LIBRARY_PATH=$LD_LIBRARY_PATH
export OLD_LIBRARY_PATH=$LIBRARY_PATH
export OLD_CUDA_HOME=$CUDA_HOME

# Save the env vars
# Add the conda/lib to LD library path to link against any .so file in there.
# Also, (and this is critical) add torch's special /lib folder as well. If we don't do this,
# Torch will link against system installed torch, which is not what we want.
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=$CONDA_PREFIX/lib:/usr/local/cuda/lib64:$LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda
