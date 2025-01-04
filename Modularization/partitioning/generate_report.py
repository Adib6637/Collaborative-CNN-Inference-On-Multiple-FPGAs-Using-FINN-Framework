from finn.transformation.fpgadataflow.annotate_cycles import AnnotateCycles
from finn.analysis.fpgadataflow.dataflow_performance import dataflow_performance
from finn.analysis.fpgadataflow.exp_cycles_per_layer import exp_cycles_per_layer
from finn.analysis.fpgadataflow.hls_synth_res_estimation import hls_synth_res_estimation
from finn.analysis.fpgadataflow.op_and_param_counts import (
    aggregate_dict_keys,
    op_and_param_counts,
)
from finn.analysis.fpgadataflow.res_estimation import (
    res_estimation,
    res_estimation_complete,
)

from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.transformation.general import RemoveUnusedTensors

def generateReport(model, target_clk_ns):
    model = model.transform(GiveUniqueNodeNames())
    synth_clk_period_ns = target_clk_ns
    ops_and_params = model.analysis(op_and_param_counts)
    estimate_layer_cycles = model.analysis(exp_cycles_per_layer)
    estimate_layer_resources = model.analysis(res_estimation)
    estimate_layer_resources["total"] = aggregate_dict_keys(estimate_layer_resources)
    estimate_layer_resources_complete = model.analysis(res_estimation_complete)  
    # need to call AnnotateCycles before dataflow_performance
    model = model.transform(AnnotateCycles())
    estimate_network_performance = model.analysis(dataflow_performance)
    # add some more metrics to estimated performance
    n_clock_cycles_per_sec = (10**9) / synth_clk_period_ns
    est_fps = n_clock_cycles_per_sec / estimate_network_performance["max_cycles"]
    estimate_network_performance["estimated_throughput_fps"] = est_fps
    est_latency_ns = (estimate_network_performance["critical_path_cycles"] * synth_clk_period_ns)
    estimate_network_performance["estimated_latency_ns"] = est_latency_ns
    return estimate_layer_resources, estimate_network_performance