#!/bin/bash

python -m pip install -e mish-cuda
pip install -r requirements.txt
apt install libgl1-mesa-glx libglib2.0-0 -y