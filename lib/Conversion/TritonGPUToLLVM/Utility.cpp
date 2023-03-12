#include "Utility.h"

namespace mlir {

namespace LLVM {
using namespace mlir::triton;

Value getStructFromElements(Location loc, ValueRange resultVals,
                            ConversionPatternRewriter &rewriter,
                            Type structType) {
  if (!structType.isa<LLVM::LLVMStructType>()) {
    return *resultVals.begin();
  }

  Value llvmStruct = rewriter.create<LLVM::UndefOp>(loc, structType);
  for (const auto &v : llvm::enumerate(resultVals)) {
    assert(v.value() && "can not insert null values");
    llvmStruct = insert_val(structType, llvmStruct, v.value(), v.index());
  }
  return llvmStruct;
}

SmallVector<Value> getElementsFromStruct(Location loc, Value llvmStruct,
                                         ConversionPatternRewriter &rewriter) {
  if (llvmStruct.getType().isIntOrIndexOrFloat() ||
      llvmStruct.getType().isa<triton::PointerType>() ||
      llvmStruct.getType().isa<LLVM::LLVMPointerType>())
    return {llvmStruct};
  ArrayRef<Type> types =
      llvmStruct.getType().cast<LLVM::LLVMStructType>().getBody();
  SmallVector<Value> results(types.size());
  for (unsigned i = 0; i < types.size(); ++i) {
    Type type = types[i];
    results[i] = extract_val(type, llvmStruct, i);
  }
  return results;
}

Value createConstantI32(Location loc, PatternRewriter &rewriter, int32_t v) {
  auto i32ty = rewriter.getIntegerType(32);
  return rewriter.create<LLVM::ConstantOp>(loc, i32ty,
                                           IntegerAttr::get(i32ty, v));
}

Value createConstantF32(Location loc, PatternRewriter &rewriter, float v) {
  auto type = type::f32Ty(rewriter.getContext());
  return rewriter.create<LLVM::ConstantOp>(loc, type,
                                           rewriter.getF32FloatAttr(v));
}

Value createConstantF64(Location loc, PatternRewriter &rewriter, float v) {
  auto type = type::f64Ty(rewriter.getContext());
  return rewriter.create<LLVM::ConstantOp>(loc, type,
                                           rewriter.getF64FloatAttr(v));
}

// Create an index type constant.
Value createIndexConstant(OpBuilder &builder, Location loc,
                          TypeConverter *converter, int64_t value) {
  Type ty = converter->convertType(builder.getIndexType());
  return builder.create<LLVM::ConstantOp>(loc, ty,
                                          builder.getIntegerAttr(ty, value));
}

// Create an integer constant of \param width bits.
Value createLLVMIntegerConstant(OpBuilder &builder, Location loc, short width,
                                int64_t value) {
  Type ty = builder.getIntegerType(width);
  return builder.create<LLVM::ConstantOp>(loc, ty,
                                          builder.getIntegerAttr(ty, value));
}

SharedMemoryObject
getSharedMemoryObjectFromStruct(Location loc, Value llvmStruct,
                                ConversionPatternRewriter &rewriter) {
  auto elems = getElementsFromStruct(loc, llvmStruct, rewriter);
  auto rank = (elems.size() - 1) / 2;
  return {/*base=*/elems[0],
          /*strides=*/{elems.begin() + 1, elems.begin() + 1 + rank},
          /*offsets=*/{elems.begin() + 1 + rank, elems.end()}};
}

SmallVector<Value>
getStridesFromShapeAndOrder(ArrayRef<int64_t> shape, ArrayRef<unsigned> order,
                            Location loc, ConversionPatternRewriter &rewriter) {
  auto rank = shape.size();
  SmallVector<Value> strides(rank);
  int64_t stride = 1;
  for (auto idx : order) {
    strides[idx] = i32_val(stride);
    stride *= shape[idx];
  }
  return strides;
}

Value storeShared(ConversionPatternRewriter &rewriter, Location loc, Value ptr,
                  Value val, Value pred) {
#if USE_ROCM
  store(val, ptr);
  return val;
#else
  MLIRContext *ctx = rewriter.getContext();
  unsigned bits = val.getType().getIntOrFloatBitWidth();
  const char *c = bits == 64 ? "l" : (bits == 16 ? "h" : "r");

  PTXBuilder builder;
  auto *ptrOpr = builder.newAddrOperand(ptr, "r");
  auto *valOpr = builder.newOperand(val, c);
  auto &st = builder.create<>("st")->shared().b(bits);
  st(ptrOpr, valOpr).predicate(pred, "b");
  return builder.launch(rewriter, loc, void_ty(ctx));
#endif
}

Value shflSync(Location loc, ConversionPatternRewriter &rewriter, Value val,
               int i) {
  unsigned bits = val.getType().getIntOrFloatBitWidth();

  if (bits == 64) {
    Type vecTy = vec_ty(f32_ty, 2);
    Value vec = bitcast(val, vecTy);
    Value val0 = extract_element(f32_ty, vec, i32_val(0));
    Value val1 = extract_element(f32_ty, vec, i32_val(1));
    val0 = shflSync(loc, rewriter, val0, i);
    val1 = shflSync(loc, rewriter, val1, i);
    vec = undef(vecTy);
    vec = insert_element(vecTy, vec, val0, i32_val(0));
    vec = insert_element(vecTy, vec, val1, i32_val(1));
    return bitcast(vec, val.getType());
  }

#ifdef USE_ROCM
  auto *context = rewriter.getContext();
  Value int_val = zext(i32_ty, bitcast(val, rewriter.getIntegerType(bits)));
  auto self_lo = rewriter.create<LLVM::CallIntrinsicOp>(UnknownLoc::get(context), TypeRange{i32_ty}, "llvm.amdgcn.mbcnt.lo", SmallVector<Value>({i32_val(-1), i32_val(0)})).getResult(0);
  auto self = rewriter.create<LLVM::CallIntrinsicOp>(UnknownLoc::get(context), TypeRange{i32_ty}, "llvm.amdgcn.mbcnt.hi", SmallVector<Value>({i32_val(-1), self_lo})).getResult(0);
  auto one = i32_val(1);
  auto two = i32_val(2);
  auto width = add(i32_val(0x3f), one);
  auto index = xor_(self, i32_val(i));
  auto self_add = add(self, width);
  auto bitnot_mask = xor_(i32_val(0x3f), i32_val(0xffffffff));
  auto upper_bound = and_(self_add, bitnot_mask);
  auto cond_cmp = icmp_sge(index, upper_bound);
  auto dst_index = select(cond_cmp, self, index);
  auto shl_index = shl(dst_index, two);
  SmallVector<Value> operands{shl_index, int_val};
  auto shfl_value = rewriter.create<LLVM::CallIntrinsicOp>(UnknownLoc::get(context), TypeRange{i32_ty}, "llvm.amdgcn.ds.permute", operands).getResult(0);
  return bitcast(trunc(rewriter.getIntegerType(bits), shfl_value), val.getType());
#else
  PTXBuilder builder;
  auto &shfl = builder.create("shfl.sync")->o("bfly").o("b32");
  auto *dOpr = builder.newOperand("=r");
  auto *aOpr = builder.newOperand(val, "r");
  auto *bOpr = builder.newConstantOperand(i);
  auto *cOpr = builder.newConstantOperand("0x1f");
  auto *maskOpr = builder.newConstantOperand("0xffffffff");
  shfl(dOpr, aOpr, bOpr, cOpr, maskOpr);
  return builder.launch(rewriter, loc, val.getType(), false);
#endif
}

} // namespace LLVM
} // namespace mlir
