from triton.language import core


# ----- FP8E4M3B15 ------
# This data-type is a variant of the standard FP8E4M3 format.
# It was designed for fast software conversion to FP16 on
# AMD GPUs that do not support it natively.
# This is the same format as FP8E4M3Nv, but:
#   - the exponent bias is 15 instead of 7
#   - 0xff and 0x7f are mapped to +-1.750 instead of +-nan
@core.builtin
def convert_fp8e4b15_to_float16(x, _builder=None):
    int8 = _builder.create_bitcast(x.handle, core.int8.to_ir(_builder))
    uint16 = _builder.create_int_cast(int8, core.uint16.to_ir(_builder), False)
    sign_mask = _builder.create_splat(_builder.get_uin16(0x08), x.shape)
    mask = _builder.create_splat(_builder.get_uin16(0xf7), x.shape)
    shift8 = _builder.create_splat(_builder.get_uin16(8), x.shape)
    shift7 = _builder.create_splat(_builder.get_uin16(7), x.shape)
    exp_man = _builder.create_and(uint16.handle, mask.handle)
    sign = _builder.create_and(uint16.handle, sign_mask.handle)
    return core.tensor(_builder.create_bitcast(
        _builder.create_or(
            _builder.create_shl(sign, shift8),
            _builder.create_shl(exp_man, shift7),
        )
    ), core.float16.to_ir(_builder))

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
