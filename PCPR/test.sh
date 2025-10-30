# Obtained from: https://github.com/lhoyer/DAFormer
# Modifications: Add HR, LR, and attention visualization
# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

#!/bin/bash

TEST_ROOT=$1
CONFIG_FILE="${TEST_ROOT}/*${TEST_ROOT: -1}.py"  # or .json for old configs
CHECKPOINT_FILE="${TEST_ROOT}/latest.pth"
SHOW_DIR="${TEST_ROOT}/preds"
echo 'Config File:' $CONFIG_FILE
echo 'Checkpoint File:' $CHECKPOINT_FILE
echo 'Predictions Output Directory:' $SHOW_DIR
python -m tools.test ${CONFIG_FILE} ${CHECKPOINT_FILE} --eval mIoU --show-dir ${SHOW_DIR} --opacity 1

 CUDA_VISIBLE_DEVICES=6 bash test.sh /home/hl/sr/mic_a/work_dirs/gtaHR2csHR_mic_hrda_650a8/gtaHR2csHR_mic_hrda_650a8.py
python -m tools.test /home/hl/sr/mic_a/work_dirs/gtaHR2csHR_mic_hrda_650a8/gtaHR2csHR_mic_hrda_650a8.py /home/hl/sr/mic_a/work_dirs/gtaHR2csHR_mic_hrda_650a8/latest.pth --eval mIoU --show-dir /home/hl/sr/mic_a/work_dirs/gtaHR2csHR_mic_hrda_650a8/preds --opacity 1
# Uncomment the following lines to visualize the LR predictions,
# HR predictions, or scale attentions of HRDA:
#python -m tools.test ${CONFIG_FILE} ${CHECKPOINT_FILE} --show-dir ${SHOW_DIR}_LR --opacity 1 --hrda-out LR
#python -m tools.test ${CONFIG_FILE} ${CHECKPOINT_FILE} --show-dir ${SHOW_DIR}_HR --opacity 1 --hrda-out HR
#python -m tools.test ${CONFIG_FILE} ${CHECKPOINT_FILE} --show-dir ${SHOW_DIR}_ATT --opacity 1 --hrda-out ATT
#python -m /home/haida/sr/mic6000/tools.test /home/haida/sr/mic6000/work_dirs/gtaHR2csHR_mic_hrda_650a8/gtaHR2csHR_mic_hrda_650a8.py /home/haida/sr/mic6000/work_dirs/gtaHR2csHR_mic_hrda_650a8/latest.pth --show-dir /home/haida/sr/mic6000/work_dirs/gtaHR2csHR_mic_hrda_650a8/ --opacity 1 --hrda-out HR