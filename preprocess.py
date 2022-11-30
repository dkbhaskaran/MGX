import os
import os.path
import argparse

import numpy as np
from PIL import Image

import migraphx

import cv2
import onnx
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("preprocess")

batchsize = 256 

def save_model(opts):
    logging.info(f'Loading model {opts.model}')
    onnx_model = onnx.load(opts.model)
    input_shape = {batchsize, 3, 224, 224}
    shape_dict = {'input_tensor:0': (batchsize, 3, 224, 224)}
    
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict, opts.data_type)
    with tvm.transform.PassContext(opt_level=3):
        factory = relay.build(mod, target='rocm -libs=miopen,rocblas', params=params)

    factory.get_lib().export_library('lib.so')
    log.info(f'Saved lib to lib.so')
    
    with open('graph.json', 'w') as f_graph_json:
        f_graph_json.write(factory.get_graph_json())    
        log.info(f'Saved graph to graph.json')

    with open('mod.params', 'wb') as f_params:
        f_params.write(relay.save_param_dict(factory.get_params()))
        log.info(f'Saved params to mod.params')

def center_crop(img, out_height, out_width):
    height, width, _ = img.shape
    left = int((width - out_width) / 2)
    right = int((width + out_width) / 2)
    top = int((height - out_height) / 2)
    bottom = int((height + out_height) / 2)
    img = img[top:bottom, left:right]
    return img

def resize_with_aspectratio(img, out_height, out_width, scale=87.5, inter_pol=cv2.INTER_LINEAR):
    height, width, _ = img.shape
    new_height = int(100. * out_height / scale)
    new_width = int(100. * out_width / scale)
    if height > width:
        w = new_width
        h = int(new_height * height / width)
    else:
        h = new_height
        w = int(new_width * width / height)
    img = cv2.resize(img, (w, h), interpolation=inter_pol)
    return img

def preprocess_image(img_path, output_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    output_height, output_width, _ = 224, 224, 3
    img = resize_with_aspectratio(img, output_height, output_width, inter_pol=cv2.INTER_AREA)
    img = center_crop(img, output_height, output_width)
    img = np.asarray(img, dtype='float32')

    means = np.array([123.68, 116.78, 103.94], dtype=np.float32)
    img -= means

    img_data = img.transpose([2, 0, 1])
    #img_data = np.expand_dims(img, axis=0)
    
    img_data = img_data.copy(order='C')
    with open(output_path, 'wb') as f:
        f.write(img_data)

def verify_saved_model():
    log.info('verifying saved model')
    with open('snake.bin', 'rb') as fp:
        img_data = np.fromfile(fp, dtype='float32').reshape(1, 3, 224, 224)

    input_name = 'input_tensor:0'
    model = migraphx.load('resnet50_v1_256_fp16.mxr', format="msgpack")

    img = np.vstack([img_data]*batchsize)
    predict_data = {input_name: img}
    pred = model.run(predict_data)

    output = np.array(pred[0])

    if output[0] - 1 == 65:
        log.info('saved model verification : success')
    else:
        log.info('saved model verification : failed')
        log.info(f'output is {tvm_output}')

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default='./resnet50_v1.onnx')
    parser.add_argument("-d", "--data-type", default='float32')
    opts = parser.parse_args()

    if not os.path.isfile('./resnet50_v1.onnx'):
        log.info(f'Downloading model')
        model_url = 'https://zenodo.org/record/2592612/files/resnet50_v1.onnx'
        download(model_url, 'resnet50_v1.onnx')
        log.info(f'Downloading model complete')

    img1_path = './imagenet_cat.png'
    if not os.path.isfile(img1_path):
        img_url = 'https://s3.amazonaws.com/model-server/inputs/kitten.jpg'
        download(img_url, './imagenet_cat.png')
    preprocess_image(img1_path, 'cat.bin')

    img2_path = './imagenet_snake.jpeg'
    if not os.path.isfile(img2_path):
        img_url = 'https://user-images.githubusercontent.com/19551872/163961172-87bc3b70-84ea-40e0-962d-be1a9d58d83d.JPEG'
        download(img_url, './imagenet_snake.jpeg')
    preprocess_image(img2_path, 'snake.bin')
    
    #save_model(opts)
    verify_saved_model()
