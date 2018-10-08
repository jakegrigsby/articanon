#!/bin/bash
cd data
python txt_to_np.py
cd ..
python train.py
python write.py --k 25 --verses 10 --chapters 8
