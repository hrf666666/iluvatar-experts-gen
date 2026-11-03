// corex_compat.h — Iluvatar CoreX 4.4.0 (CUDA 10.2 API surface) vs modern CUDA gaps.
// Force-included by the ivcc wrapper for .cu sources and by the Strata build's
// CXX flags for host sources that call the CUDA runtime directly.
//
// v2 note (kept for the record): the old byval half2 pointer-shim macros were
//   removed — at -O1 the intrinsics inline natively and the macros miscompiled
//   (raw f16 bits returned as floats). -O1 itself is required: at -O0 the
//   non-inlined device calls become RPCMOV/PCSWAP sequences that raise XID-22
//   illegal-instruction interrupts on BI-V150 and wedge the GPU channel.
//
// v3: the CUDA 11/12/13 API surface that Strata uses, mapped onto CoreX 10.2.
#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda.h>

// ---- cudaStreamWaitEvent: CoreX headers only declare the 3-argument form ----
#if !defined(COREX_COMPAT_STREAMWAITEVENT)
#define COREX_COMPAT_STREAMWAITEVENT
inline cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event) {
    return cudaStreamWaitEvent(stream, event, 0);
}
#endif

// ---- 64-bit warp shuffles: `unsigned long long` hangs the kernel on bi and
//      the others are unverified; route every 64-bit shuffle through two
//      32-bit shuffles (verified bit-exact). Macro dispatch avoids overload
//      ambiguity with the header templates; 4-arg width forms forward to the
//      real builtins (32-bit types only in those call sites). ----------------
#if defined(__CUDACC__)
// width=32 forwarding primitives. The bi hardware warp is 64 lanes, so the
// DEFAULT shuffle width is 64: an absolute-lane `__shfl_sync(m, v, 0)`
// broadcast reads lane 0 of the 64-lane warp and gives every OTHER 32-lane
// group that group's value (verified: adjacent 32-groups collapse pairwise).
// The 4-argument width=32 form is exact, so EVERY 3-arg shuffle is routed
// through it.
__device__ __forceinline__ unsigned bi_shfl_up_u32w(unsigned m, unsigned v, int d, int w) { return __shfl_up_sync(m, v, d, w); }
__device__ __forceinline__ unsigned bi_shfl_down_u32w(unsigned m, unsigned v, int d, int w) { return __shfl_down_sync(m, v, d, w); }
__device__ __forceinline__ unsigned bi_shfl_xor_u32w(unsigned m, unsigned v, int d, int w) { return __shfl_xor_sync(m, v, d, w); }
__device__ __forceinline__ unsigned bi_shfl_idx_u32w(unsigned m, unsigned v, int s, int w) { return __shfl_sync(m, v, s, w); }
__device__ __forceinline__ unsigned bi_shfl_up_u32(unsigned m, unsigned v, int d) { return bi_shfl_up_u32w(m, v, d, 32); }
__device__ __forceinline__ unsigned bi_shfl_down_u32(unsigned m, unsigned v, int d) { return bi_shfl_down_u32w(m, v, d, 32); }
__device__ __forceinline__ unsigned bi_shfl_xor_u32(unsigned m, unsigned v, int d) { return bi_shfl_xor_u32w(m, v, d, 32); }
__device__ __forceinline__ unsigned bi_shfl_idx_u32(unsigned m, unsigned v, int s) { return bi_shfl_idx_u32w(m, v, s, 32); }
__device__ __forceinline__ unsigned long long bi_p64(unsigned hi, unsigned lo) { return ((unsigned long long) hi << 32) | lo; }

// 3-arg overloads
__device__ __forceinline__ unsigned long long bi_shfl_up_sync(unsigned m, unsigned long long v, int d) { return bi_p64(bi_shfl_up_u32(m, (unsigned) (v >> 32), d), bi_shfl_up_u32(m, (unsigned) v, d)); }
__device__ __forceinline__ unsigned long long bi_shfl_down_sync(unsigned m, unsigned long long v, int d) { return bi_p64(bi_shfl_down_u32(m, (unsigned) (v >> 32), d), bi_shfl_down_u32(m, (unsigned) v, d)); }
__device__ __forceinline__ unsigned long long bi_shfl_xor_sync(unsigned m, unsigned long long v, int d) { return bi_p64(bi_shfl_xor_u32(m, (unsigned) (v >> 32), d), bi_shfl_xor_u32(m, (unsigned) v, d)); }
__device__ __forceinline__ unsigned long long bi_shfl_sync(unsigned m, unsigned long long v, int s) { return bi_p64(bi_shfl_idx_u32(m, (unsigned) (v >> 32), s), bi_shfl_idx_u32(m, (unsigned) v, s)); }
__device__ __forceinline__ float bi_shfl_up_sync(unsigned m, float v, int d) { return __int_as_float(bi_shfl_up_u32(m, __float_as_int(v), d)); }
__device__ __forceinline__ float bi_shfl_down_sync(unsigned m, float v, int d) { return __int_as_float(bi_shfl_down_u32(m, __float_as_int(v), d)); }
__device__ __forceinline__ float bi_shfl_xor_sync(unsigned m, float v, int d) { return __int_as_float(bi_shfl_xor_u32(m, __float_as_int(v), d)); }
__device__ __forceinline__ float bi_shfl_sync(unsigned m, float v, int s) { return __int_as_float(bi_shfl_idx_u32(m, __float_as_int(v), s)); }
__device__ __forceinline__ int bi_shfl_up_sync(unsigned m, int v, int d) { return (int) bi_shfl_up_u32(m, (unsigned) v, d); }
__device__ __forceinline__ int bi_shfl_down_sync(unsigned m, int v, int d) { return (int) bi_shfl_down_u32(m, (unsigned) v, d); }
__device__ __forceinline__ int bi_shfl_xor_sync(unsigned m, int v, int d) { return (int) bi_shfl_xor_u32(m, (unsigned) v, d); }
__device__ __forceinline__ int bi_shfl_sync(unsigned m, int v, int s) { return (int) bi_shfl_idx_u32(m, (unsigned) v, s); }
__device__ __forceinline__ unsigned bi_shfl_up_sync(unsigned m, unsigned v, int d) { return bi_shfl_up_u32(m, v, d); }
__device__ __forceinline__ unsigned bi_shfl_down_sync(unsigned m, unsigned v, int d) { return bi_shfl_down_u32(m, v, d); }
__device__ __forceinline__ unsigned bi_shfl_xor_sync(unsigned m, unsigned v, int d) { return bi_shfl_xor_u32(m, v, d); }
__device__ __forceinline__ unsigned bi_shfl_sync(unsigned m, unsigned v, int s) { return bi_shfl_idx_u32(m, v, s); }

__device__ __forceinline__ double bi_shfl_up_sync(unsigned m, double v, int d) { unsigned long long r = bi_shfl_up_sync(m, *reinterpret_cast<unsigned long long*>(&v), d); double f; *reinterpret_cast<unsigned long long*>(&f) = r; return f; }
__device__ __forceinline__ double bi_shfl_down_sync(unsigned m, double v, int d) { unsigned long long r = bi_shfl_down_sync(m, *reinterpret_cast<unsigned long long*>(&v), d); double f; *reinterpret_cast<unsigned long long*>(&f) = r; return f; }
__device__ __forceinline__ double bi_shfl_xor_sync(unsigned m, double v, int d) { unsigned long long r = bi_shfl_xor_sync(m, *reinterpret_cast<unsigned long long*>(&v), d); double f; *reinterpret_cast<unsigned long long*>(&f) = r; return f; }
__device__ __forceinline__ double bi_shfl_sync(unsigned m, double v, int s) { unsigned long long r = bi_shfl_sync(m, *reinterpret_cast<unsigned long long*>(&v), s); double f; *reinterpret_cast<unsigned long long*>(&f) = r; return f; }
__device__ __forceinline__ long long bi_shfl_up_sync(unsigned m, long long v, int d) { return (long long) bi_shfl_up_sync(m, (unsigned long long) v, d); }
__device__ __forceinline__ long long bi_shfl_down_sync(unsigned m, long long v, int d) { return (long long) bi_shfl_down_sync(m, (unsigned long long) v, d); }
__device__ __forceinline__ long long bi_shfl_xor_sync(unsigned m, long long v, int d) { return (long long) bi_shfl_xor_sync(m, (unsigned long long) v, d); }
__device__ __forceinline__ long long bi_shfl_sync(unsigned m, long long v, int s) { return (long long) bi_shfl_sync(m, (unsigned long long) v, s); }
__device__ __forceinline__ unsigned long bi_shfl_up_sync(unsigned m, unsigned long v, int d) { return (unsigned long) bi_shfl_up_sync(m, (unsigned long long) v, d); }
__device__ __forceinline__ unsigned long bi_shfl_down_sync(unsigned m, unsigned long v, int d) { return (unsigned long) bi_shfl_down_sync(m, (unsigned long long) v, d); }
__device__ __forceinline__ unsigned long bi_shfl_xor_sync(unsigned m, unsigned long v, int d) { return (unsigned long) bi_shfl_xor_sync(m, (unsigned long long) v, d); }
__device__ __forceinline__ unsigned long bi_shfl_sync(unsigned m, unsigned long v, int s) { return (unsigned long) bi_shfl_sync(m, (unsigned long long) v, s); }
__device__ __forceinline__ long bi_shfl_up_sync(unsigned m, long v, int d) { return (long) bi_shfl_up_sync(m, (unsigned long long) v, d); }
__device__ __forceinline__ long bi_shfl_down_sync(unsigned m, long v, int d) { return (long) bi_shfl_down_sync(m, (unsigned long long) v, d); }
__device__ __forceinline__ long bi_shfl_xor_sync(unsigned m, long v, int d) { return (long) bi_shfl_xor_sync(m, (unsigned long long) v, d); }
__device__ __forceinline__ long bi_shfl_sync(unsigned m, long v, int s) { return (long) bi_shfl_sync(m, (unsigned long long) v, s); }

// 4-arg (width) overloads: 32-bit types forward to the builtins with the width
__device__ __forceinline__ float bi_shfl_down_sync(unsigned m, float v, int d, int w) { return __int_as_float(bi_shfl_down_u32w(m, __float_as_int(v), d, w)); }
__device__ __forceinline__ float bi_shfl_xor_sync(unsigned m, float v, int d, int w) { return __int_as_float(bi_shfl_xor_u32w(m, __float_as_int(v), d, w)); }
__device__ __forceinline__ float bi_shfl_up_sync(unsigned m, float v, int d, int w) { return __int_as_float(bi_shfl_up_u32w(m, __float_as_int(v), d, w)); }
__device__ __forceinline__ int bi_shfl_down_sync(unsigned m, int v, int d, int w) { return (int) bi_shfl_down_u32w(m, (unsigned) v, d, w); }
__device__ __forceinline__ int bi_shfl_xor_sync(unsigned m, int v, int d, int w) { return (int) bi_shfl_xor_u32w(m, (unsigned) v, d, w); }
__device__ __forceinline__ int bi_shfl_up_sync(unsigned m, int v, int d, int w) { return (int) bi_shfl_up_u32w(m, (unsigned) v, d, w); }
__device__ __forceinline__ unsigned bi_shfl_down_sync(unsigned m, unsigned v, int d, int w) { return bi_shfl_down_u32w(m, v, d, w); }
__device__ __forceinline__ unsigned bi_shfl_xor_sync(unsigned m, unsigned v, int d, int w) { return bi_shfl_xor_u32w(m, v, d, w); }

__device__ __forceinline__ int bi_shfl_sync(unsigned m, int v, int s, int w) { return (int) bi_shfl_idx_u32w(m, (unsigned) v, s, w); }
__device__ __forceinline__ unsigned bi_shfl_sync(unsigned m, unsigned v, int s, int w) { return bi_shfl_idx_u32w(m, v, s, w); }
__device__ __forceinline__ float bi_shfl_sync(unsigned m, float v, int s, int w) { return __int_as_float(bi_shfl_idx_u32w(m, __float_as_int(v), s, w)); }

// dispatch (variadic: 3-arg and 4-arg call sites). Scoped to Strata builds:
// llama.cpp's ggml-cuda uses further types whose overloads are not modeled
// here, and its own kernels verified working without the interception.
#ifdef STRATA_CUDA_ARCH_NUM
#define __shfl_up_sync(...)     bi_shfl_up_sync(__VA_ARGS__)
#define __shfl_down_sync(...)   bi_shfl_down_sync(__VA_ARGS__)
#define __shfl_xor_sync(...)    bi_shfl_xor_sync(__VA_ARGS__)
#define __shfl_sync(...)        bi_shfl_sync(__VA_ARGS__)
#endif // STRATA_CUDA_ARCH_NUM
#endif // defined(__CUDACC__)

// ---- __ldcs for scalar types: CoreX 4.4 only overloads half/bf16 --------------
//      (5.0's sm_32_intrinsics.h declares the full scalar set — ivcc passes
//      -DCOREX_GE_50=1 there and the shims are dropped to avoid redefinitions)
#if defined(__CUDACC__) && !defined(COREX_GE_50)
__device__ __forceinline__ int   __ldcs(const int* p)   { return *(const volatile int*) p; }
__device__ __forceinline__ unsigned int __ldcs(const unsigned int* p) { return *(const volatile unsigned int*) p; }
__device__ __forceinline__ float __ldcs(const float* p) { return *(const volatile float*) p; }
__device__ __forceinline__ short __ldcs(const short* p) { return *(const volatile short*) p; }
__device__ __forceinline__ unsigned short __ldcs(const unsigned short* p) { return *(const volatile unsigned short*) p; }
__device__ __forceinline__ char  __ldcs(const char* p)  { return *(const volatile char*) p; }
__device__ __forceinline__ unsigned char __ldcs(const unsigned char* p) { return *(const volatile unsigned char*) p; }
__device__ __forceinline__ long long __ldcs(const long long* p) { return *(const volatile long long*) p; }
__device__ __forceinline__ unsigned long long __ldcs(const unsigned long long* p) { return *(const volatile unsigned long long*) p; }
__device__ __forceinline__ int2  __ldcs(const int2* p)  { int2 r; r.x = ((const volatile int*) p)[0]; r.y = ((const volatile int*) p)[1]; return r; }
__device__ __forceinline__ uint2 __ldcs(const uint2* p) { uint2 r; r.x = ((const volatile unsigned*) p)[0]; r.y = ((const volatile unsigned*) p)[1]; return r; }
__device__ __forceinline__ float2 __ldcs(const float2* p) { float2 r; r.x = ((const volatile float*) p)[0]; r.y = ((const volatile float*) p)[1]; return r; }
#endif // defined(__CUDACC__)

// ---- device attributes (CUDA 11.x+) ----------------------------------------
#define cudaDevAttrReservedSharedMemoryPerBlock cudaDevAttrMaxSharedMemoryPerBlock
#define CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED

// ---- driver entry points (CUDA 12.0+): map to the plain 11.0 form ---------
//      (5.0 declares the real ixDriverEntryPointQueryResult surface natively)
#if !defined(COREX_GE_50)
typedef int cudaDriverEntryPointQueryResult;   // CoreX 10.2 has none; an int compares like the enum
#define cudaEnableDefault 0
#define cudaDriverEntryPointSuccess 0
inline __host__ cudaError_t cudaGetDriverEntryPointByVersion(const char* name, void** pp, int, unsigned,
                                                    cudaDriverEntryPointQueryResult* q) {
    if (q) *q = cudaDriverEntryPointSuccess;
    return cudaGetDriverEntryPoint(name, pp, 0ull);   // CoreX 10.2: (name, ptr, flags)
}
// Strata's own pre-12.5 fallback passes (name, ptr, flags, result)
inline __host__ cudaError_t cudaGetDriverEntryPoint(const char* name, void** pp, unsigned,
                                                    cudaDriverEntryPointQueryResult* q) {
    if (q) *q = cudaDriverEntryPointSuccess;
    return cudaGetDriverEntryPoint(name, pp, 0ull);
}
#endif // !COREX_GE_50

// ---- graphs (CUDA 12.x): v2 adds per-edge data; drop it --------------------
#define cudaGraphGetEdges_v2(g, f, t, e, n) cudaGraphGetEdges(g, f, t, n)

// CUDA 13's 3-arg cudaGraphInstantiate -> the legacy 5-arg CoreX form
inline __host__ cudaError_t cudaGraphInstantiate(cudaGraphExec_t* pExec, cudaGraph_t graph, unsigned long long) {
    return cudaGraphInstantiate(pExec, graph, (cudaGraphNode_t*) nullptr, (char*) nullptr, (size_t) 0);
}

// ---- capture info v3 (CUDA 12.3+): dependencies/edge data not available ----
inline __host__ cudaError_t cudaStreamGetCaptureInfo_v3(cudaStream_t stream, cudaStreamCaptureStatus* status,
                                               cudaGraph_t* graph, const void*, const void*, size_t* numDeps) {
    if (numDeps) *numDeps = 0;
    unsigned long long id = 0;
    return cudaStreamGetCaptureInfo_v2(stream, status, &id, graph, nullptr, numDeps);
}

// ---- host register flags: ReadOnly (12.4+) is an optimization, drop it -----
#define cudaHostRegisterReadOnly 0

// ---- cudaInitDevice (12.0+): set device + flags the 10.2 way --------------
inline cudaError_t cudaInitDevice(int device, unsigned int deviceFlags, unsigned int) {
    cudaError_t e = cudaSetDevice(device);
    if (e == cudaSuccess) e = cudaSetDeviceFlags(deviceFlags);
    return e;
}

// ---- thread block clusters (Hopper+): not on CoreX; report one "cluster" --
#define cudaOccupancyMaxActiveClusters(out, fn, cfg) (*(out) = 1, cudaSuccess)

// ---- batched memcpy (CUDA 12.8+, STRATA_DMA_BATCH): fall back to the error
//      path so the callers use the per-copy loop ------------------------------
//      (5.0 ships ixMemcpyAttributes + the cudaMemcpyAttributes mapping macro;
//      our struct would collide — gated out there)
#if !defined(COREX_GE_50)
struct cudaMemcpyAttributes {
    void* src;
    void* dst;
    size_t size;
};
#define cudaMemcpyFlagPreferOverlapWithCompute 0
inline cudaError_t cudaMemcpyBatchAsync(const cudaMemcpyAttributes*, size_t, size_t*, unsigned) {
    return cudaErrorNotSupported;   // dma_batch falls back to sequential copies
}
#endif // !COREX_GE_50
