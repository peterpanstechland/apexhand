#!/usr/bin/env bash
# Isolate from ROS Humble / conda Python so Isaac Sim 6.0.1 + Python 3.12 stay clean.
unset PYTHONPATH
unset AMENT_PREFIX_PATH
unset COLCON_PREFIX_PATH
unset CMAKE_PREFIX_PATH
unset LD_PRELOAD
# Conda's older libstdc++ is picked up via the venv's base_prefix (miniconda).
# Kit then fails: GLIBCXX_3.4.30 not found. Force the Ubuntu 22.04 runtime.
unset LD_LIBRARY_PATH
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64"
export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libstdc++.so.6"

source "$HOME/isaacsim-env/bin/activate"

export OMNI_KIT_ACCEPT_EULA=YES
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json
export ISAACLAB_PATH="$HOME/Documents/apexhand/IsaacLab"
export PIP_CONSTRAINT="$HOME/Documents/apexhand/constraints.txt"
export PYTHONPATH="$HOME/Documents/apexhand/source${PYTHONPATH:+:$PYTHONPATH}"
