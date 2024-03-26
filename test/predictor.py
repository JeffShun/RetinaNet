from calendar import c
from dis import dis
from os.path import abspath, dirname
from typing import IO, Dict

import numpy as np
import torch
import yaml

from train.config.model_config import network_cfg
import tensorrt as trt
import pycuda.driver as pdd
import pycuda.autoinit

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

class PredictConfig:

    def __init__(self, test_cfg):
        # 配置文件
        self.input_size = test_cfg.get('input_size')


    def __repr__(self) -> str:
        return str(self.__dict__)


class PredictModel:

    def __init__(self, model_f: IO, config_f):
        # TODO: 模型文件定制
        self.model_f = model_f 
        self.config_f = config_f
        self.network_cfg = network_cfg


class Predictor:

    def __init__(self, device: str, model: PredictModel):
        self.device = torch.device(device)
        self.model = model
        self.tensorrt_flag = False 

        with open(self.model.config_f, 'r') as config_f:
            self.test_cfg = PredictConfig(yaml.safe_load(config_f))
        self.network_cfg = model.network_cfg
        self.load_model()

    def load_model(self) -> None:
        if isinstance(self.model.model_f, str):
            # 根据后缀判断类型
            if self.model.model_f.endswith('.pth'):
                self.load_model_pth()
            elif self.model.model_f.endswith('.pt'):
                self.load_model_jit()
            elif self.model.model_f.endswith('.engine'):
                self.tensorrt_flag = True
                self.load_model_engine()

    def load_model_jit(self) -> None:
        # 加载静态图
        from torch import jit
        self.net = jit.load(self.model.model_f, map_location=self.device)
        self.net.eval()
        self.net.to(self.device).half()

    def load_model_pth(self) -> None:
        # 加载动态图
        self.net = self.network_cfg.network
        checkpoint = torch.load(self.model.model_f, map_location=self.device)
        self.net.load_state_dict(checkpoint)
        self.net.eval()
        self.net.to(self.device).half()

    def load_model_engine(self) -> None:
        TRT_LOGGER = trt.Logger()
        runtime = trt.Runtime(TRT_LOGGER)
        with open(self.model.model_f, 'rb') as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

    def allocate_buffers(self, engine, context):
        inputs = []
        outputs = []
        bindings = []
        stream = pdd.Stream()
        for i, binding in enumerate(engine):
            size = trt.volume(context.get_binding_shape(i))
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = pdd.pagelocked_empty(size, dtype)
            device_mem = pdd.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(device_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        return inputs, outputs, bindings, stream

    def trt_inference(self, context, bindings, inputs, outputs, stream, batch_size):
        # Transfer input data to the GPU.
        [pdd.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
        # Run inference.
        context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        [pdd.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
        # Synchronize the stream
        stream.synchronize()
        # Return only the host outputs.
        return [out.host for out in outputs]

    def pre_process(self, img):
        img = (img-img.min())/(img.max()-img.min())
        img_t = torch.from_numpy(img).float()
        resize_image = torch.nn.functional.interpolate(img_t[None, None], size=self.test_cfg.input_size, mode="bilinear")
        return resize_image

    def post_porcess(self, scores, labels, bboxes, ori_h, ori_w):
        image_height, image_width = self.test_cfg.input_size
        props = []
        for box, label, score in zip(bboxes, labels, scores):
            if score < 0:
                continue
            box = box.cpu().numpy()
            cen_x, cen_y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            w, h = box[2] - box[0], box[3] - box[1]
            w, h = w / 2, h / 2

            prop = [
                int(np.round((cen_x - w) / image_width * ori_w)),
                int(np.round((cen_y - h) / image_height * ori_h)),
                int(np.round((cen_x + w) / image_width * ori_w)),
                int(np.round((cen_y + h) / image_height * ori_h)),
                int(label.cpu().numpy()),
                float(score.cpu().numpy()),
                ]
            
            props.append(prop)
        return props

    def predict(self, img):
        ori_h, ori_w = img.shape
        img_t = self.pre_process(img)
        scores, labels, bboxes = self.net(img_t.half().to(self.device))
        props = self.post_porcess(scores[0], labels[0], bboxes[0], ori_h, ori_w)
        return props



