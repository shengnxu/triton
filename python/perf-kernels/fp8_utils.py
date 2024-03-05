import math
import torch

# should use e4m3 for forward pass and e5m2 for backward pass
# set intended dtype and scaling factor if fp8
fp8_max_repr_val_dict = {}
fp8_type_list = []

TORCH_HAS_FP8E4FNUZ = hasattr(torch, 'float8_e4m3fnuz')
TORCH_HAS_FP8E4FN = hasattr(torch, 'float8_e4m3fn')
TORCH_HAS_FP8E5FNUZ = hasattr(torch, 'float8_e5m2fnuz')
TORCH_HAS_FP8E5 = hasattr(torch, 'float8_e5m2')
if TORCH_HAS_FP8E4FNUZ:
    fp8_max_repr_val_dict[torch.float8_e4m3fnuz] = 240.0
    fp8_type_list.append(torch.float8_e4m3fnuz)
if TORCH_HAS_FP8E4FN:
    fp8_max_repr_val_dict[torch.float8_e4m3fn] = 448.0
    fp8_type_list.append(torch.float8_e4m3fn)
if TORCH_HAS_FP8E5FNUZ:
    fp8_max_repr_val_dict[torch.float8_e5m2fnuz] = 57344.0
    fp8_type_list.append(torch.float8_e5m2fnuz)
if TORCH_HAS_FP8E5:
    fp8_max_repr_val_dict[torch.float8_e5m2] = 57344.0
    fp8_type_list.append(torch.float8_e5m2)

def _tensor_amax(tensor: torch.Tensor) -> float:
    # use .item() because only python float, rather than pytorch float tensor
    # can be recoginized as fp32 in triton
    return tensor.abs().float().amax().item()

def get_fp8_scaling_factor(tensor: torch.Tensor,
                                fp8_type: torch.dtype,
                                margin: float = 1.0) -> float:
    assert fp8_type in fp8_type_list, f"Intended dtype {fp8_type} not supported by PyTorch"
    fp8_max_repr_val = fp8_max_repr_val_dict[fp8_type]
    ret = fp8_max_repr_val / _tensor_amax(tensor)
    ret = math.pow(2, math.floor(math.log2(ret))-margin)
    return ret

def scale_and_cast_to_fp8(
        tensor: torch.Tensor,
        fp8_type: torch.dtype = torch.float8_e4m3fnuz,
        scaling_factor: float = None,
        margin: float = 1.0,
    ) -> tuple[torch.Tensor, float, float]:
    """Scale a non-fp8 tensor to fp8 and return scaling and descaling factor

    Args:
        tensor (torch.Tensor): non-fp8 tensor to be scaled and casted
        fp8_type (torch.dtype, optional): fp8 type to be casted to. Defaults to torch.float8_e4m3fnuz.
        scaling_factor (float, optional): If None, will use tensor.amax()
        margin (float, optional): margin to compute scaling factor. Defaults to 1.0.

    Returns:
        tuple[torch.Tensor, float, float]: scaled fp8 tensor, scale, descale
    """
    """

    Args:
        tensor (torch.Tensor):
        fp8_type (torch.dtype): _description_
        scaling_factor (float, optional): _description_. Defaults to None.

    Returns:
        tuple[torch.Tensor, float, float]: _description_
    """
    assert fp8_type in fp8_type_list, f"Intended dtype {fp8_type} not supported by PyTorch"
    # get the maximally representable number for this fp8 type
    fp8_max_repr_val = fp8_max_repr_val_dict[fp8_type]
    # use tensor's amax or user input
    amax = _tensor_amax(tensor) if scaling_factor is None else scaling_factor
    scale = fp8_max_repr_val / amax
    # rounding to nearest lower power of 2
    scale = math.pow(2, math.floor(math.log2(scale)) - margin)
    # scale and cast
    tensor_scaled_fp8 = (tensor.float() * scale).to(fp8_type)

    return tensor_scaled_fp8, scale, 1.0/scale