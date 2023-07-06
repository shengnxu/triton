#include "Utility.h"
#include "TypeConverter.h"

namespace mlir {

namespace LLVM {
using namespace mlir::triton;

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
  ArrayRef<Type> types =
      llvmStruct.getType().cast<LLVM::LLVMStructType>().getBody();
  SmallVector<Value> elems(types.size());
  for (unsigned i = 0; i < types.size(); ++i) {
    Type type = types[i];
    elems[i] = extract_val(type, llvmStruct, i);
  }

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
  // This map facilates the butterfly shuffle pattern for a stride less
  // than 16. The pattern stride is the key of the map.
  GCNBuilder builder;
  // TODO: Investigate if it is possible to use dpp_ctrl field in the instruction
  // ss << "dpp_ctrl:0x" << std::hex << dpp_ctrl << ' ';
  if (i == 0){
    auto shfl = builder.create("ds_permute_b32");
    auto dOpr = builder.newOperand(val, "v");
    auto buffOpr = builder.newOperand(i32_val(0), "v");
    auto offsetOpr = builder.newConstantOperand(std::string("offset:252\n"));
    (*shfl)(dOpr, buffOpr, dOpr, offsetOpr);
    auto swait = builder.create("s_waitcnt lgkmcnt(0)\n");
    (*swait)();
    return builder.launch(rewriter, loc, val.getType(), false);
  }
  DenseMap<short, std::string> dpp_ctrl{
	            {32, "row_bcast:31"}, 
		    {16, "row_bcast:15"}, 
		    {8, "row_shr:8"}, 
		    {4, "row_shr:4"}, 
		    {2, "quad_perm:[2, 3, 0, 1]"}, 
		    {1, "qual_perm:[1, 0, 3, 2]"}};
  auto shfl = builder.create("v_move_b32");
  auto dOpr = builder.newOperand("=v");
  auto aOpr = builder.newOperand(val, "v");
  auto ctrlOpr = builder.newConstantOperand(dpp_ctrl[i]);
  auto rowMaskOpr = builder.newConstantOperand(std::string("row_mask:0xf "));
  auto bankMaskOpr = builder.newConstantOperand(std::string("bank_mask:0xf "));
  auto boundCtrlOpr = builder.newConstantOperand(std::string("bound_ctrl:0\n"));
  (*shfl)(dOpr, aOpr, ctrlOpr, rowMaskOpr, bankMaskOpr, boundCtrlOpr);
  auto swait = builder.create("s_waitcnt lgkmcnt(0)\n");
  (*swait)();
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

Value addStringToModule(Location loc, ConversionPatternRewriter &rewriter,
                        StringRef key, StringRef content) {
  auto moduleOp = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  auto ctx = moduleOp.getContext();
  unsigned stringNumber = 0;
  SmallString<16> stringConstName;
  do {
    stringConstName.clear();
    (key + Twine(stringNumber++)).toStringRef(stringConstName);
  } while (moduleOp.lookupSymbol(stringConstName));

  llvm::SmallString<64> contentStr(content);
  size_t contentSize = contentStr.size_in_bytes();
  auto globalType = LLVM::LLVMArrayType::get(i8_ty, contentSize);

  LLVM::GlobalOp global;
  {
    ConversionPatternRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());
    global = rewriter.create<LLVM::GlobalOp>(
        UnknownLoc::get(ctx), globalType,
        /*isConstant=*/true, LLVM::Linkage::Internal, stringConstName,
        rewriter.getStringAttr(contentStr));
  }

  Value zero = i32_val(0);
  Value globalPtr =
      rewriter.create<LLVM::AddressOfOp>(UnknownLoc::get(ctx), global);
  Value stringStart =
      rewriter.create<LLVM::GEPOp>(UnknownLoc::get(ctx), ptr_ty(i8_ty),
                                   globalPtr, SmallVector<Value>({zero, zero}));
  return stringStart;
}

} // namespace LLVM
} // namespace mlir
