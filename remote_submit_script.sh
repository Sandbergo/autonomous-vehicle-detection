#!/bin/bash

set -e

# Usage:
# bash remote_submit_script <username> <password> <iteration_of_model.pth>

if ! [ -x "$(command -v sshpass)" ]; then
    echo 'sshpass is not installed, updating before installing'
    # sudo pacman -S sshpass
    # sudo apt-get update
    # sudo apt-get install sshpass
fi

#Setting up clab strings
USER=$1
PASSWORD=$2
IT_NUMBER=$3
MODEL_PTH="model_0"$IT_NUMBER".pth"
MODEL_COMPUTER="$USER@clab15.idi.ntnu.no"
INFERENCE_COMPUTER="$USER@clab11.idi.ntnu.no"

echo "$USER"
echo "$PASSWORD"
echo "$IT_NUMBER"
echo "$MODEL_PTH"
echo "$MODEL_COMPUTER"
echo "$INFERENCE_COMPUTER"

# Copying model_XXX.pth file from clab15 to clab11
echo 'copying model file from 15 to 11'
sshpass -p $PASSWORD scp \
        $MODEL_COMPUTER:/work/"$USER"/autonomous-vehicle-detection/SSD/outputs/resnet_960x720_nonUpsample/"$MODEL_PTH"  \
        $INFERENCE_COMPUTER:/work/"$USER"/autonomous-vehicle-detection/SSD/outputs/resnet_960x720_nonUpsample/"$MODEL_PTH"

# Copying last_checkpoint.txt file from clab15 to clab11
echo 'copying last_checkpoint file from 15 to 11'
sshpass -p $PASSWORD scp \
        $MODEL_COMPUTER:/work/"$USER"/autonomous-vehicle-detection/SSD/outputs/resnet_960x720_nonUpsample/last_checkpoint.txt  \
        $INFERENCE_COMPUTER:/work/"$USER"/autonomous-vehicle-detection/SSD/outputs/resnet_960x720_nonUpsample/last_checkpoint.txt

# Running submit_results
echo 'running submit results'
sshpass -p $PASSWORD ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no $INFERENCE_COMPUTER /bin/bash <<EOF
        cd /work/$USER/autonomous-vehicle-detection/SSD/
        . ~/.bashrc
        conda activate TermProject
        python3 submit_results.py configs/resnet_waymo_920x720_nonUpsample.yaml

        cd outputs/
        cp -r resnet_960x720_nonUpsample resnet_960x720_nonUpsample_"$IT_NUMBER"
        rm resnet_960x720_nonUpsample/*
EOF

# Transfer result back to user machine
echo 'copying .json to host machine'
sshpass -p $PASSWORD scp \
        $INFERENCE_COMPUTER:/work/"$USER"/autonomous-vehicle-detection/SSD/outputs/resnet_960x720_nonUpsample_"$IT_NUMBER"/test_detected_boxes.json  \
        ~/Documents/DatasynExtra/test_detected_boxes_920x720_"$IT_NUMBER".json
