# Darknet to ONNX Converter for NeuroWeave SDK

![](https://img.shields.io/static/v1?label=python&message=3.9|3.10&color=blue)
![](https://img.shields.io/static/v1?label=pytorch&message=2.0&color=<COLOR>)
[![](https://img.shields.io/static/v1?label=license&message=APACHE2&color=green)](./LICENSE.txt)

```
# Folder Structure
.
├── LICENSE.txt
├── README.md
├── __ini__.py
├── config.py
├── darknet2onnx.py # Main file to execute the conversion from darknet to onnx
├── darknet2pytorch.py
├── region_loss.py
├── torch_utils.py
├── utils.py
├── yolo_layer.py
```

# Conversion from Darknet 2 ONNX
## Modifications made for compatiblity with NeuroWeave SDK
- Removes the final post processing YOLO Head from the model
- Makes the output as the feature outputs expected by NW-SDK
- Expand operation over tensor used 6D tensors, which are not
  compatible with NW-SDK, hence replaced the custom implementation
  of upsample op with nn.Upsample official implementation in torch
  for upsampling the tensor.
- Matches the output with Upsample_interpolate and Darkflow's op
  for upsample.
- NW-SDK's post-processing subgraph uses image features as NHWC
  whereas pytorch results in NCHW features, added a transpose in
  the end to create NHWC outputs - this transpose should be optimized
  away in the compilation of the model
- Maps Reorg implementation in PyTorch which uses 6D reshapes and
  transposes to perform Space to Depth operation with ONNX's native
  SpaceToDepth op. (Tried PyTorch's PixelUnsqueeze which is numerically
  not equivalent)

## How to run in Docker

The included Dockerfile can build a container which includes all dependencies. By default it runs the `python darknet2onnx.py` and takes the rest of the arguments from the command line.

1. Build and test with:

```sh
docker build -t darknet2onnx .
docker run darknet2onnx --help
```

2. Run export command with the model directory attached as a mounted volume:

```sh
# Assuming the local model directory is `./dnn_model`
docker run -v ./dnn_model:/dnn_model darknet2onnx --batch_size 1 --onnx_file_path /dnn_model/yolov4.onnx /dnn_model/yolov4.cfg /dnn_model/yolov4.weights
```

## Steps to run manually
1. Install all standard dependencies of XNNC 3.x which includes python 3.10 and pytorch 2.0
2. Download weights and cfgs of YOLO v2/v3/v4 models (or their variants)
3. Execute following command to execute conversion from darknet cfg/weights to onnx model
    - `python darknet2onnx.py <path-to-cfg> <path-to-weights> --batch_size 1 --onnx_file_path <path-to-onnx-model>`
    - For example: `python darknet2onnx.py darknet/cfg/yolov3-spp.cfg darknet/weights/yolov3-spp.weights --batch_size 1 --onnx_file_path yolov3-spp.onnx`

# Reference & Citations
- https://github.com/AlexeyAB/darknet
- https://github.com/Tianxiaomo/pytorch-YOLOv4
- https://github.com/eriklindernoren/PyTorch-YOLOv3
- https://github.com/marvis/pytorch-caffe-darknet-convert
- https://github.com/marvis/pytorch-yolo3

```
@misc{bochkovskiy2020yolov4,
      title={YOLOv4: Optimal Speed and Accuracy of Object Detection},
      author={Alexey Bochkovskiy and Chien-Yao Wang and Hong-Yuan Mark Liao},
      year={2020},
      eprint={2004.10934},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
```
@InProceedings{Wang_2021_CVPR,
    author    = {Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark},
    title     = {{Scaled-YOLOv4}: Scaling Cross Stage Partial Network},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {13029-13038}
}
```
