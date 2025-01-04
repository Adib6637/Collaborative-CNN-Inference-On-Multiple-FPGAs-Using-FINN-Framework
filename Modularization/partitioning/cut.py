from qonnx.core.modelwrapper import ModelWrapper
from copy import deepcopy
from qonnx.util.basic import qonnx_make_model
from qonnx.transformation.remove import remove_node_and_rewire
import uuid
default_build_dir = "../../../../notebooks/GitHub/M_Project/Patitioning/tmp"

def generate_block(model_folded_path, start=None, end=None, build_dir=default_build_dir):

    if start == None:
        start = 0
    if end == None:
        end = start+1

    if end < start :
        print("invalid start and end combination. return None")
        return None

    model = ModelWrapper(model_folded_path)
    last_output_shape = model.get_tensor_shape(model.graph.node[end-1].output[0])
    last_output_dtype = model.get_tensor_datatype(model.graph.node[end-1].output[0]) 

    
    first_input_shape = model.get_tensor_shape(model.graph.node[start].input[0])
    first_input_dtype = model.get_tensor_datatype(model.graph.node[start].input[0]) 
    
    back_nodes = model.graph.node[:start]
    front_nodes = model.graph.node[end:]
    
    for node in back_nodes:
        remove_node_and_rewire(model, node)
    for node in front_nodes:
        remove_node_and_rewire(model, node)
        
    path_one_layer_model = build_dir + f"/model_tmp_cut_{str(uuid.uuid4())}.onnx"
    model.save(path_one_layer_model)
    return path_one_layer_model, last_output_shape, last_output_dtype, first_input_shape, first_input_dtype