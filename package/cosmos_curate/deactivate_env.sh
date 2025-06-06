#!/bin/sh
# This is run when a conda env is deactivated. See activate_env.sh for more info.

export LD_LIBRARY_PATH=$OLD_LD_LIBRARY_PATH
unset OLD_LD_LIBRARY_PATH
export LIBRARY_PATH=$OLD_LIBRARY_PATH
unset OLD_LIBRARY_PATH
export CUDA_HOME=$OLD_CUDA_HOME
unset OLD_CUDA_HOME