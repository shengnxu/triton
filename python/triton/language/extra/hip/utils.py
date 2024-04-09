from triton.language import core


# ----- FP8E4M3B15 ------
# This data-type is a variant of the standard FP8E4M3 format.
# It was designed for fast software conversion to FP16 on
# AMD GPUs that do not support it natively.
# This is the same format as FP8E4M3Nv, but:
#   - the exponent bias is 15 instead of 7
#   - 0xff and 0x7f are mapped to +-1.750 instead of +-nan
@core.builtin
def convert_fp8e4b15_to_float16(x: core.tensor, _builder=None):
    # bitcast the fp8e4b15 to uint16
    x = x.to(core.uint8, bitcast=True, _builder=_builder)
    x = x.to(core.uint16, _builder=_builder)
    # get sign and exponent + mantissa individually
    num_mask = core.tensor(_builder.get_uint16((1 << 7) - 1), core.uint16)
    num = x.__and__(num_mask, _builder=_builder)
    sign = x.__rshift__(core.tensor(_builder.get_uint16(7), core.uint16), _builder=_builder)
    # left shift signa and exponent + mantissa
    sign = sign.__lshift__(core.tensor(_builder.get_uint16(15), core.uint16), _builder=_builder)
    num = num.__lshift__(core.tensor(_builder.get_uint16(7), core.uint16), _builder=_builder)
    # cast back to float16
    y = num.__or__(sign, _builder=_builder)
    y = y.to(core.float16, bitcast=True, _builder=_builder)
    return y


@core.builtin
def convert_custom_float8(arg, dst_ty, fp_downcast_rounding, _builder=None):
    assert arg.type.scalar.is_fp8e4b15()
    upcast_val = convert_fp8e4b15_to_float16(arg, _builder=_builder)
    if dst_ty.scalar.is_fp32():
        upcast_val = upcast_val.to(core.float32, _builder=_builder)
    return upcast_val

    # assert arg.type.scalar.is_fp16() or arg.type.scalar.is_fp32()
    # downcast_val = arg
    # if arg.type.scalar.is_fp32():
    #     downcast_val = downcast_val.to(core.float16, fp_downcast_rounding="rtz", _builder=_builder)
    # downcast_val = convert_float16_to_fp8e4b15(downcast_val, _builder=_builder)
    # return downcast_val
