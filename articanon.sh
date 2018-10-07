#!/bin/bash
python data/txt_to_np.py
python train.py
python write.py --k 25 --verses 10 --chapters 8
