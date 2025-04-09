import time
import dcn_cuda_backward
import dcn_cuda_forward

import math
import torch
from typing import Any
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.cuda.amp import custom_fwd, custom_bwd
from .forward import forward_kernel


class DCNFunction(Function):
    BP_FUNCS = [
        dcn_cuda_backward.backward_p1_c2_tile16_thread128,
        dcn_cuda_backward.backward_p1_c4_tile16_thread128,
        dcn_cuda_backward.backward_p2_c2_tile16_thread128,
        dcn_cuda_backward.backward_p1_c2_tile16_thread256,
        dcn_cuda_backward.backward_p1_c4_tile16_thread256,
        dcn_cuda_backward.backward_p2_c2_tile16_thread256,
        dcn_cuda_backward.backward_p1_c2_tile16_thread384,
        dcn_cuda_backward.backward_p1_c4_tile16_thread384,
        dcn_cuda_backward.backward_p2_c2_tile16_thread384,
        dcn_cuda_backward.backward_p1_c2_tile16_thread512,
        dcn_cuda_backward.backward_p1_c4_tile16_thread512,
        dcn_cuda_backward.backward_p2_c2_tile16_thread512,
        dcn_cuda_backward.backward_p1_c2_tile16_thread768,
        dcn_cuda_backward.backward_p1_c4_tile16_thread768,
        dcn_cuda_backward.backward_p2_c2_tile16_thread768,
        dcn_cuda_backward.backward_p1_c2_tile32_thread128,
        dcn_cuda_backward.backward_p1_c2_tile32_thread256,
        dcn_cuda_backward.backward_p1_c2_tile32_thread384,
        dcn_cuda_backward.backward_p1_c2_tile32_thread512,
    ]
    FW_FUNCS = [
        dcn_cuda_forward.dcn_forward_l256_c4,
        dcn_cuda_forward.dcn_forward_l256_c8,
        dcn_cuda_forward.dcn_forward_l256_c16,
    ]
    BP_TABLES = dict()
    FW_TABLES = dict()


    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx: Any, values, deformables, weights) -> Any:
        B, H, W, G, C = values.shape
        func = DCNFunction.find_fw_funcs(values, deformables, weights)
        out = torch.zeros_like(values)
        func(B, G, C, H, W, values, deformables, weights, out)
        return out

    @staticmethod
    def find_fw_funcs(values, deformables, weights):
        B, H, W, G, C = values.shape
        B, H, W, G, K = weights.shape
        hash_value = 10000 * B + 100 * H + W + 1000 * G
        if hash_value in DCNFunction.FW_TABLES.keys():
            return DCNFunction.FW_TABLES[hash_value]
        print("missing")
        candicate_func = None
        min_t = 999.0
        outs = torch.zeros_like(values)
        for func in DCNFunction.FW_FUNCS:
            t = []
            for i in range(100):
                torch.cuda.synchronize()
                start_t = time.time()
                func(B, G, C, H, W, values, deformables, weights, outs)
                torch.cuda.synchronize()
                t.append(time.time() - start_t)
            t = t[-50:]
            t = sum(t) / len(t)
            if t < min_t:
                min_t = t
                DCNFunction.FW_TABLES[hash_value] = func
                candicate_func = func
        assert candicate_func is not None
        print(candicate_func)
        return candicate_func
    @staticmethod
    def find_bp_funcs(values, deformables, weights, grad_out):
        B, H, W, G, C = values.shape
        B, H, W, G, K = weights.shape
        hash_value = 10000 * B + 100 * H + W + 1000 * G
        if hash_value in DCNFunction.BP_TABLES.keys():
            return DCNFunction.BP_TABLES[hash_value]
        print("missing")
        candicate_func = None
        min_t = 999.0
        grad_values = torch.zeros_like(values)
        grad_deformables = torch.zeros_like(deformables)
        grad_weights = torch.zeros_like(weights)
        for func in DCNFunction.BP_FUNCS:
            t = []
            for i in range(100):
                torch.cuda.synchronize()
                start_t = time.time()
                func(B, H, W, G, K, C, values, deformables, weights, grad_out, grad_values, grad_deformables, grad_weights)
                torch.cuda.synchronize()
                t.append(time.time() - start_t)
            t = t[-50:]
            t = sum(t) / len(t)
            if t < min_t:
                min_t = t
                DCNFunction.BP_TABLES[hash_value] = func
                candicate_func = func
        assert candicate_func is not None
        print(candicate_func)
        return candicate_func

    @staticmethod
    @once_differentiable
    @custom_bwd
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        grad_out = grad_outputs[0]
        values, deformables, weights = ctx.saved_tensors
        B, H, W, G, C = values.shape
        B, H, W, G, K = weights.shape
        func = DCNFunction.find_bp_funcs(values, deformables, weights, grad_out)
        grad_values = torch.zeros_like(values)
        grad_deformables = torch.zeros_like(deformables)
        grad_weights = torch.zeros_like(weights)
        func(B, H, W, G, K, C, values, deformables, weights, grad_out, grad_values, grad_deformables, grad_weights)
        return grad_values, grad_deformables, grad_weights