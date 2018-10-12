#!/bin/bash
cd data
python txt_to_np.py
cd ..
python train.py
python write.py --k 15 --verses 12 --chapters 40 --filter False
