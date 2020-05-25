#!/bin/bash

# Usage:
# bash clab_available.sh <username> <password>

if ! [ -x "$(command -v sshpass)" ]; then
    echo 'sshpass is not installed, updating before installing'
    # sudo pacman -S sshpass
    sudo apt-get update
    sudo apt-get install sshpass
fi

#Password
PASSWORD=$2

#Setting up clab strings
USER=$1
CLAB_UNDER_10="@clab0"
CLAB_OVER_10="@clab"
LINK=".idi.ntnu.no"

for clab_i in {1..24}
do
    if [ $clab_i -lt 10 ]
    then
        CLAB_COMPUTER=$USER$CLAB_UNDER_10$clab_i$LINK
    else
        CLAB_COMPUTER=$USER$CLAB_OVER_10$clab_i$LINK
    fi
    echo -n "GPU Memory of $CLAB_COMPUTER: "
    sshpass -p $PASSWORD ssh -o ConnectTimeout=3 -o StrictHostKeyChecking=no $CLAB_COMPUTER \
            '(nvidia-smi --query-gpu=memory.used --format=csv)' >&1
done

