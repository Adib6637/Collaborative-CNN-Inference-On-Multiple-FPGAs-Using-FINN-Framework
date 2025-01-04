from qonnx.core.modelwrapper import ModelWrapper
from brevitas.export import export_qonnx
from qonnx.util.cleanup import cleanup as qonnx_cleanup
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
import uuid


def prepareModel(model_name_, build_dir_, model_raw_path_):
    model_name = model_name_
    model_raw_path = f"{model_raw_path_}/{model_name}.onnx"
    model_cleaned_path = f"{build_dir_}/cnv_1bit_trained_cleaned_{str(uuid.uuid4())}.onnx"
    model_finn_path = f"{build_dir_}/cnv_1bit_trained_finn_{str(uuid.uuid4())}.onnx"
    # clean up the model
    qonnx_cleanup(model_raw_path, out_file=model_cleaned_path)
    model = ModelWrapper(model_cleaned_path)
    # transform to FINN format
    model = model.transform(ConvertQONNXtoFINN())
    model.save(model_finn_path)
    return model_finn_path