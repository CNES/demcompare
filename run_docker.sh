#!/bin/sh

docker run \
  --rm \
  --ipc=host \
  -it \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /home/chthen/Documents/CNES_RT_Surfaces_3D/code:/code \
  rtsurface3d_docker /bin/bash
