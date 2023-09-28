#!/bin/bash


docker run \
    -it \
    -d \
    --rm \
    --ipc host \
    --gpus all \
    --shm-size 14G \
    --device /dev/video0:/dev/video0 \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY=$DISPLAY \
    -p 7860:7860 \
    -v $(pwd):/nanoowl \
    nanoowl:23-01