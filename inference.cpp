#include <chrono>
#include <fstream>
#include <mutex>
#include <stdlib.h>
#include <string>
#include <thread>

#include <glog/logging.h>
#include <hip/hip_runtime_api.h>
#include <migraphx/migraphx.hpp>

std::mutex GlobalLock;
const size_t batchsize = 2;

class MiGraphXObject {
public:
  explicit MiGraphXObject(const std::string &ModelPath, int DevId)
      : DeviceId(DevId) {
    std::lock_guard<std::mutex> guard(GlobalLock);
    auto status = hipSetDevice(DevId);
    LOG(INFO) << "Setting the environment variable for device " << DevId;
    CHECK(setenv("ROCR_VISIBLE_DEVICES", std::to_string(DevId).c_str(), true) ==
          0)
        << "Setting the environment variable failed";
    LOG(INFO) << "Setting the environment variable completed for " << DevId;

#ifndef PRECOMPILED_MODEL
    migraphx::onnx_options onnx_opts;
    Prog = migraphx::parse_onnx(ModelPath.c_str(), onnx_opts);
#else
    Prog = migraphx::load(ModelPath.c_str());
#endif

    LOG(INFO) << "Parsing ONNX model for device id " << DeviceId << std::endl;
    if (VLOG_IS_ON(1)) {
      Prog.print();
    }

#ifndef PRECOMPILED_MODEL // Only required if compilation is needed
    migraphx::target targ = migraphx::target("gpu");

    migraphx::compile_options compOpts;
    compOpts.set_offload_copy();

    LOG(INFO) << DeviceId << "::Compiling program for gpu...";
    Prog.compile(targ, compOpts);
    if (VLOG_IS_ON(1)) {
      Prog.print();
    }
#endif

    ShapeParams = Prog.get_parameter_shapes();
  }

  migraphx::api::program_parameter_shapes &getShape() { return ShapeParams; }
  migraphx::program &getProg() { return Prog; }

  int getDeviceId() { return DeviceId; }

private:
  int DeviceId;
  migraphx::program Prog;
  migraphx::api::program_parameter_shapes ShapeParams;
};

bool Infer(MiGraphXObject &Obj, const std::string InputPath) {
  std::ifstream input(InputPath, std::ios::binary);
  std::string inData((std::istreambuf_iterator<char>(input)),
                     std::istreambuf_iterator<char>());

  auto status = hipSetDevice(Obj.getDeviceId());
  auto Prog = Obj.getProg();
  auto shapeParams = Obj.getShape();
  auto inputName = shapeParams.names().front();

  migraphx::program_parameters params;
  params.add(inputName,
             migraphx::argument(shapeParams[inputName], inData.data()));

  LOG(INFO) << Obj.getDeviceId() << "::Starting inference";
  auto start = std::chrono::high_resolution_clock::now();
  auto outputs = Prog.eval(params);
  auto stop = std::chrono::high_resolution_clock::now();
  auto elapsed =
      std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

  LOG(INFO) << Obj.getDeviceId() << "::Inference completed in "
            << elapsed.count() * 1e-3 << "ms" << std::endl;

  auto shape = outputs[0].get_shape();
  auto lengths = shape.lengths();

  int64_t *results = reinterpret_cast<int64_t *>(outputs[0].data());
  LOG(INFO) << Obj.getDeviceId() << "::the image is " << results[0];
  return true;
}

class thread_obj {
public:
  void operator()(const std::string &ModelPath, const std::string ImagePath,
                  const int DeviceId) {
    MiGraphXObject Obj(ModelPath, DeviceId);
    Infer(Obj, ImagePath);
  }
};

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);

  CHECK(argc != 3) << "Usage inference model.onnx image_path";
  std::string model = argv[1];
  std::string image1 = argv[2];
  std::string image2 = argv[3];

  std::thread th1(thread_obj(), model, image1, 3 /* Device ID */);
  std::thread th2(thread_obj(), model, image2, 4 /* Device ID */);
  th1.join();
  th2.join();
}
