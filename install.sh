sudo apt install ffmpeg
cd app/modules/easydict
python3 setup.py install
cd ../../..
wget https://raw.githubusercontent.com/Sharpiless/Yolov5-Deepsort/main/deep_sort/deep_sort/deep/checkpoint/ckpt.t7 -P models/deepsort
