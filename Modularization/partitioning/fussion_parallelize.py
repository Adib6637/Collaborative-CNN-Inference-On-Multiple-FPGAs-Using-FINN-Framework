from finn.util.basic import make_build_dir
from finn.util.visualization import showInNetron
import torch
import onnx
from brevitas.export import export_qonnx
from qonnx.util.cleanup import cleanup as qonnx_cleanup
from qonnx.core.modelwrapper import ModelWrapper
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames, RemoveStaticGraphInputs
from finn.util.pytorch import ToTensor
from qonnx.transformation.merge_onnx_models import MergeONNXModels
from qonnx.core.datatype import DataType
from qonnx.transformation.insert_topk import InsertTopK
from qonnx.transformation.infer_datatypes import InferDataTypes
from finn.transformation.streamline import Streamline
from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul
from qonnx.transformation.bipolar_to_xnor import ConvertBipolarMatMulToXnorPopcount
import finn.transformation.streamline.absorb as absorb
from finn.transformation.streamline.reorder import MakeMaxPoolNHWC, MoveScalarLinearPastInvariants
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.transformation.general import RemoveUnusedTensors
import finn.transformation.fpgadataflow.convert_to_hls_layers as to_hls
from finn.transformation.fpgadataflow.create_dataflow_partition import (
    CreateDataflowPartition,
)
from finn.transformation.move_reshape import RemoveCNVtoFCFlatten
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from finn.util.basic import alveo_part_map
from finn.transformation.fpgadataflow.vitis_build import VitisBuild
from finn.transformation.fpgadataflow.make_pynq_driver import MakePYNQDriver
from qonnx.custom_op.registry import getCustomOp
from finn.transformation.fpgadataflow.set_folding import SetFolding
from finn.transformation.fpgadataflow.set_fifo_depths import InsertAndSetFIFODepths

def fussionParallelize(build_dir_input, model_name_input):
    build_dir = build_dir_input
    model_name = model_name_input
    model_path = f"{build_dir_input}/{model_name_input}"
    
    model = ModelWrapper(model_path)
    model = model.transform(ConvertQONNXtoFINN())
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(RemoveStaticGraphInputs())
    
    global_inp_name = model.graph.input[0].name
    ishape = model.get_tensor_shape(global_inp_name)
    # preprocessing: torchvision's ToTensor divides uint8 inputs by 255
    totensor_pyt = ToTensor()
    chkpt_preproc_name = build_dir+f"/{model_name}_preproc.onnx"
    export_qonnx(totensor_pyt, torch.randn(ishape), chkpt_preproc_name)
    qonnx_cleanup(chkpt_preproc_name, out_file=chkpt_preproc_name)
    pre_model = ModelWrapper(chkpt_preproc_name)
    pre_model = pre_model.transform(ConvertQONNXtoFINN())
    
    # join preprocessing and core model
    model = model.transform(MergeONNXModels(pre_model))
    # add input quantization annotation: UINT8 for all BNN-PYNQ models
    global_inp_name = model.graph.input[0].name
    model.set_tensor_datatype(global_inp_name, DataType["UINT8"])
    
    # postprocessing: insert Top-1 node at the end
    model = model.transform(InsertTopK(k=1))
    chkpt_name = build_dir+f"/{model_name}_pre_post.onnx"
    # tidy-up again
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferDataTypes())
    model = model.transform(RemoveStaticGraphInputs())
    model.save(chkpt_name)
    
    model = ModelWrapper(build_dir + f"/{model_name}_pre_post.onnx")
    #model = model.transform(MoveScalarLinearPastInvariants())
    model = model.transform(absorb.AbsorbSignBiasIntoMultiThreshold()) #n
    model = model.transform(Streamline())
    model = model.transform(LowerConvsToMatMul())
    model = model.transform(MakeMaxPoolNHWC())
    model = model.transform(absorb.AbsorbTransposeIntoMultiThreshold())
    model = model.transform(ConvertBipolarMatMulToXnorPopcount())
    model = model.transform(Streamline())
    # absorb final add-mul nodes into TopK
    model = model.transform(absorb.AbsorbScalarMulAddIntoTopK())
    model = model.transform(InferDataLayouts())
    model = model.transform(RemoveUnusedTensors())
    model.save(build_dir + f"/{model_name}_streamlined.onnx")
    
    # choose the memory mode for the MVTU units, decoupled or const
    mem_mode = "decoupled"
    
    model = ModelWrapper(build_dir + f"/{model_name}_streamlined.onnx")
    #model = ModelWrapper(f"/scratch/hpc-prf-ekiapp/sheikh/finn_new/finn-on-n2/finn/notebooks/advanced/output_estimates_only/intermediate_models/step_convert_to_hls.onnx")
    
    model = model.transform(to_hls.InferBinaryMatrixVectorActivation(mem_mode))
    model = model.transform(to_hls.InferQuantizedMatrixVectorActivation(mem_mode))
    # TopK to LabelSelect
    model = model.transform(to_hls.InferLabelSelectLayer())
    # input quantization (if any) to standalone thresholding
    model = model.transform(to_hls.InferThresholdingLayer())
    model = model.transform(to_hls.InferConvInpGen())
    model = model.transform(to_hls.InferStreamingMaxPool())
    # get rid of Reshape(-1, 1) operation between hlslib nodes
    model = model.transform(RemoveCNVtoFCFlatten())
    # get rid of Tranpose -> Tranpose identity seq
    model = model.transform(absorb.AbsorbConsecutiveTransposes())
    # infer tensor data layouts
    model = model.transform(InferDataLayouts())
    parent_model = model.transform(CreateDataflowPartition())
    parent_model.save(build_dir + f"/{model_name}_dataflow_parent.onnx")
    try:
        sdp_node = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")[0]
        sdp_node = getCustomOp(sdp_node)
        dataflow_model_filename = sdp_node.get_nodeattr("model")
        # save the dataflow partition with a different name for easier access
        dataflow_model = ModelWrapper(dataflow_model_filename)
        dataflow_model.save(build_dir + f"/{model_name}_dataflow_model.onnx")
    except:
        print("no StreamingDataflowPartition")
        print("stop")
        return None
    
    model = ModelWrapper(build_dir + f"/{model_name}_dataflow_model.onnx")
    
    fc_layers = model.get_nodes_by_op_type("MatrixVectorActivation")
    # each tuple is (PE, SIMD, in_fifo_depth) for a layer
    folding = [
        (16, 3, [128]),
        (32, 32, [128]),
        (16, 32, [128]),
        (16, 32, [128]),
        (4, 32, [81]),
        (1, 32, [2]),
        (1, 4, [2]),
        (1, 8, [128]),
        (2, 1, [3]),
    ]
    for fcl, (pe, simd, ififodepth) in zip(fc_layers, folding):
        fcl_inst = getCustomOp(fcl)
        fcl_inst.set_nodeattr("PE", pe)
        fcl_inst.set_nodeattr("SIMD", simd)
        fcl_inst.set_nodeattr("inFIFODepths", ififodepth)
    
    # use same SIMD values for the sliding window operators
    swg_layers = model.get_nodes_by_op_type("ConvolutionInputGenerator")
    for i in range(len(swg_layers)):
        swg_inst = getCustomOp(swg_layers[i])
        simd = folding[i][1]
        swg_inst.set_nodeattr("SIMD", simd)
    
    test_pynq_board = "Pynq-Z1"
    target_clk_ns = 10
    
    model_folded_path = build_dir + f"/{model_name}_folded.onnx"
    
    model = model.transform(GiveUniqueNodeNames())
    model.save(model_folded_path)
    return model_folded_path