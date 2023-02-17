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
    llvmStruct = insert_val(structType, llvmStruct, v.value(),
                            rewriter.getI64ArrayAttr(v.index()));
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
    results[i] = extract_val(type, llvmStruct, i64_arr_attr(i));
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
  // This map facilates the butterfly shuffle pattern for a stride less than 16. The pattern stride is the key of the map.
  GCNBuilder builder;
  switch (i){
    case 32:
    {
      auto cOpr0 = builder.newConstantOperand(0);
      auto cOprN1 = builder.newConstantOperand(-1);
      auto cOpr2 = builder.newConstantOperand(2);
      auto cOpr32 = builder.newConstantOperand(32);
      auto lower = builder.create("v_mbcnt_lo_u32_b32");
      auto lowerResOpr = builder.newOperand("=v");
      auto lowerOpr1 = builder.newOperand(cOprN1->value, "v");
      auto lowerOpr2 = builder.newOperand(cOpr0->value, "v");
      (*lower)(lowerResOpr, lowerOpr1, lowerOpr2);
      lowerResOpr->constraint = "v";
      auto higher = builder.create("v_mbcnt_hi_u32_b32");
      auto higherResOpr = builder.newOperand("=v");
      (*higher)(higherResOpr, lowerOpr1, lowerResOpr);
      higherResOpr->constraint = "v";
      auto vor = builder.create("v_or_b32_e32");
      auto vorResOpr = builder.newOperand("=v");
      (*vor)(vorResOpr, cOpr32, higherResOpr);
      vorResOpr->constraint = "v";
      auto cmp = builder.create("v_cmp_gt_i32_e32 vcc 64");
      (*cmp)(vorResOpr);
      auto vmask = builder.create("v_cndmask_b32_e32");
      auto vmaskResOpr = builder.newOperand("=v");
      auto restOpr = builder.newConstantOperand("vcc");
      (*vmask)(vmaskResOpr, higherResOpr, vorResOpr, restOpr);
      vmaskResOpr->constraint = "v";
      auto multiplier = builder.create("v_lshlrev_b32_e32");
      auto mulResOpr = builder.newOperand("=v");
      auto bitShift = builder.newOperand(cOpr2->value, "v");
      (*multiplier)(mulResOpr, bitShift, vmaskResOpr);
      mulResOpr->constraint = "v";
      auto swait0 = builder.create("s_waitcnt vmcnt(0)");
      (*swait0)();
      auto permute = builder.create("ds_bpermute_b32");
      auto dOpr = builder.newOperand("=v");
      auto aOpr = builder.newOperand(val, "v");
      (*permute)(dOpr, mulResOpr, aOpr);
      auto swait1 = builder.create("s_waitcnt lgkmcnt(0)");
      (*swait1)();
      break;
    }
    case 16:
    case 8:
    case 4:
    case 2:
    case 1:
    {
      DenseMap<short, unsigned int> masks{{16, 0x401F}, {8, 0x201F}, {4, 0x101F}, {2, 0x081F}, {1, 0x041F}};
      auto shfl = builder.create("ds_swizzle_b32");
      auto dOpr = builder.newOperand("=v");
      auto aOpr = builder.newOperand(val, "v");
      auto maskOpr = builder.newConstantOperand("offset:" + std::to_string(masks[i]));
      (*shfl)(dOpr, aOpr, maskOpr);
      auto swait = builder.create("s_waitcnt lgkmcnt(0)");
      (*swait)();
      break;
    }
    default:
    {
      llvm::report_fatal_error("Unsupported warpSize");
    }
  }
#else
  PTXBuilder builder;
  auto &shfl = builder.create("shfl.sync")->o("bfly").o("b32");
  auto *dOpr = builder.newOperand("=r");
  auto *aOpr = builder.newOperand(val, "r");
  auto *bOpr = builder.newConstantOperand(i);
  auto *cOpr = builder.newConstantOperand("0x1f");
  auto *maskOpr = builder.newConstantOperand("0xffffffff");
  shfl(dOpr, aOpr, bOpr, cOpr, maskOpr);
#endif
  return builder.launch(rewriter, loc, val.getType(), false);
}

} // namespace LLVM
} // namespace mlir
