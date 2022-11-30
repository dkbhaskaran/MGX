# Prerequisite

1. ROCM 5.3.0
2. libgoogle-glog-dev

# Build
```
mkdir build
cd build; cmake ../; make; cd ..
```

# Run
```
python preprocess.py
GLOG_logtostderr=1 ./build/inference ./resnet50_v1.onnx ./snake.bin ./cat.bin
```
