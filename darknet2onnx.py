import functools
import sys
import torch
from darknet2pytorch import Darknet
import torch.onnx.symbolic_helper as sym_help
import torch.onnx.symbolic_opset11 as opset11
from torch.onnx.symbolic_helper import parse_args, _unimplemented
from torch.onnx._internal import jit_utils, registration

_onnx_symbolic = functools.partial(registration.onnx_symbolic, opset=11)

@_onnx_symbolic("darknet::reorg")
@parse_args("v", "i")
def onnx_space_to_depth(g: jit_utils.GraphContext, self, downscale_factor):
    rank = sym_help._get_tensor_rank(self)
    if rank is not None and rank != 4:
        return sym_help._unimplemented("darknet_reorg", "only support 4d input")
    return g.op("SpaceToDepth", self, blocksize_i=downscale_factor)

opset11.darknet_reorg = onnx_space_to_depth

def transform_to_onnx(cfgfile, weightfile, batch_size=1, onnx_file_name=None):
    model = Darknet(cfgfile)

    model.print_network()
    model.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    dynamic = False
    if batch_size <= 0:
        dynamic = True

    input_names = ["input"]

    if dynamic:
        x = torch.randn((1, 3, model.height, model.width), requires_grad=True)
        out = model(x)
        output_names = [f"feat_{i}"for i in range(len(out))]
        if not onnx_file_name:
            onnx_file_name = "yolo_{}_{}_dynamic.onnx".format(model.height, model.width)
        dynamic_axes = {"input": {0: "batch_size"}, "boxes": {0: "batch_size"}, "confs": {0: "batch_size"}}
        # Export the model
        print('Export the onnx model ...')
        torch.onnx.export(model,
                          x,
                          onnx_file_name,
                          export_params=True,
                          opset_version=11,
                          do_constant_folding=True,
                          input_names=input_names, output_names=output_names,
                          dynamic_axes=dynamic_axes)
    else:
        x = torch.randn((batch_size, 3, model.height, model.width), requires_grad=True)
        out = model(x)
        output_names = [f"feat_{i}"for i in range(len(out))]
        if not onnx_file_name:
            onnx_file_name = "yolo_{}_3_{}_{}_static.onnx".format(batch_size, model.height, model.width)
        torch.onnx.export(model,
                          x,
                          onnx_file_name,
                          export_params=True,
                          opset_version=11,
                          do_constant_folding=True,
                          input_names=input_names, output_names=output_names,
                          dynamic_axes=None)

    print(f'Onnx model exporting done at: {onnx_file_name}')
    return onnx_file_name


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('weightfile')
    parser.add_argument('--batch_size', type=int, help="Static Batchsize of the model. use batch_size<=0 for dynamic batch size")
    parser.add_argument('--onnx_file_path', help="Output onnx file path")
    args = parser.parse_args()
    transform_to_onnx(args.config, args.weightfile, args.batch_size, args.onnx_file_path)

