# Prerequisite
ROCM 5.3.0
Google loggin

# Build
mkdir build
cd build; cmake ../; make; cd ..

# Run
python preprocess.py
GLOG_logtostderr=1 ./build/inference ./resnet50_v1.onnx ./snake.bin ./cat.bin
