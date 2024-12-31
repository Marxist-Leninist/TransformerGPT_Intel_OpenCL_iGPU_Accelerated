#!/usr/bin/env python3
"""
opencl_backend.py

OpenCL-based GPU ops for matrix multiplication, add_bias, ReLU, and rowwise softmax.
Used by other modules in this 'my_transformers' folder.

Dependencies:
  pip install pyopencl numpy

Example usage:
  from my_transformers.opencl_backend import OpenCLBackend
  backend = OpenCLBackend()
  out = backend.gpu_matmul(A_np, B_np)

If you run this file directly:
  python opencl_backend.py
it will do a small self-test.
"""

import pyopencl as cl
import numpy as np

##############################################################################
# Inline OpenCL Kernel Code
##############################################################################
KERNEL_CODE = r"""
#pragma OPENCL EXTENSION cl_khr_fp32 : enable

//----------------------------------------
// 1) matmul_2d
//    A: (M,K), B: (K,N) => C: (M,N)
//----------------------------------------
__kernel void matmul_2d(
    __global const float* A,
    __global const float* B,
    __global float* C,
    int M,
    int K,
    int N
)
{
    int row = get_global_id(0);
    int col = get_global_id(1);

    if (row < M && col < N)
    {
        float val = 0.0f;
        for (int i = 0; i < K; i++){
            val += A[row*K + i] * B[i*N + col];
        }
        C[row*N + col] = val;
    }
}

//----------------------------------------
// 2) add_bias
//    X: (M,N), bias: (N), out: (M,N)
//----------------------------------------
__kernel void add_bias(
    __global const float* X,
    __global const float* bias,
    int M,
    int N,
    __global float* out
)
{
    int row = get_global_id(0);
    int col = get_global_id(1);

    if (row < M && col < N)
    {
        int idx = row*N + col;
        out[idx] = X[idx] + bias[col];
    }
}

//----------------------------------------
// 3) relu_inplace
//    Flatten shape => total_elems
//    X: shape (total_elems)
//----------------------------------------
__kernel void relu_inplace(
    __global float* X,
    int total_elems
)
{
    int idx = get_global_id(0);
    if (idx < total_elems)
    {
        float val = X[idx];
        if (val < 0.0f)
        {
            X[idx] = 0.0f;
        }
    }
}

//----------------------------------------
// 4) softmax_2d
//    rowwise softmax on shape (B,T)
//----------------------------------------
__kernel void softmax_2d(
    __global float* X,
    int B,
    int T
)
{
    int row = get_global_id(0);

    if (row < B)
    {
        float maxval = -1e30f;
        for (int i=0; i<T; i++)
        {
            float v = X[row*T + i];
            if (v > maxval) maxval = v;
        }
        float sum_exp = 0.0f;
        for (int i=0; i<T; i++)
        {
            float e = exp(X[row*T + i] - maxval);
            sum_exp += e;
        }
        for (int i=0; i<T; i++)
        {
            float e = exp(X[row*T + i] - maxval);
            X[row*T + i] = e / (sum_exp + 1e-9f);
        }
    }
}
"""


class OpenCLBackend:
    """
    A minimal OpenCL backend providing GPU-based:
      - matmul
      - add_bias
      - relu_inplace
      - rowwise softmax

    Called by other modules (multi_head_attention, feed_forward, etc.)
    """

    def __init__(self, local_size=(16,16)):
        """
        local_size: (int,int) controlling kernel local work size. 
        """
        platforms = cl.get_platforms()
        selected_device = None
        for plat in platforms:
            gpus = plat.get_devices(device_type=cl.device_type.GPU)
            if gpus:
                selected_device = gpus[0]
                break
        if selected_device is None:
            selected_device = platforms[0].get_devices()[0]
            print("[Warning] Using CPU device:", selected_device.name)
        else:
            print("[Info] Using GPU device:", selected_device.name)

        self.ctx = cl.Context([selected_device])
        self.queue = cl.CommandQueue(self.ctx)
        self.local_size = local_size

        # Build from inline code
        self.program = cl.Program(self.ctx, KERNEL_CODE).build()

    def gpu_matmul(self, A_np: np.ndarray, B_np: np.ndarray) -> np.ndarray:
        """
        A: (M,K), B: (K,N) => out: (M,N)
        """
        M,K = A_np.shape
        K2,N= B_np.shape
        if K!=K2:
            raise ValueError("gpu_matmul shape mismatch")

        mf = cl.mem_flags
        A_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A_np)
        B_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B_np)
        C_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, size=M*N*4)

        global_size = (
            int(np.ceil(M/self.local_size[0])*self.local_size[0]),
            int(np.ceil(N/self.local_size[1])*self.local_size[1])
        )
        evt = self.program.matmul_2d(
            self.queue,
            global_size,
            self.local_size,
            A_buf, B_buf, C_buf,
            np.int32(M), np.int32(K), np.int32(N)
        )
        evt.wait()

        out_np = np.empty((M,N), dtype=np.float32)
        cl.enqueue_copy(self.queue, out_np, C_buf).wait()
        return out_np

    def gpu_add_bias(self, X_np: np.ndarray, bias_np: np.ndarray) -> np.ndarray:
        """
        X: shape (M,N)
        bias: shape (N,)
        => out: (M,N)
        """
        M,N = X_np.shape
        if bias_np.shape[0]!=N:
            raise ValueError("gpu_add_bias mismatch")

        mf = cl.mem_flags
        X_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=X_np)
        b_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=bias_np)
        O_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, size=M*N*4)

        global_size = (
            int(np.ceil(M/self.local_size[0])*self.local_size[0]),
            int(np.ceil(N/self.local_size[1])*self.local_size[1])
        )
        evt = self.program.add_bias(
            self.queue,
            global_size,
            self.local_size,
            X_buf, b_buf,
            np.int32(M), np.int32(N),
            O_buf
        )
        evt.wait()

        out = np.empty((M,N), dtype=np.float32)
        cl.enqueue_copy(self.queue, out, O_buf).wait()
        return out

    def gpu_relu_inplace(self, X_np: np.ndarray) -> np.ndarray:
        """
        In-place ReLU: flatten X_np, run kernel, reshape back.
        """
        flat = X_np.ravel()
        total = flat.size

        mf = cl.mem_flags
        buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=flat)
        evt = self.program.relu_inplace(
            self.queue,
            (total,),
            None,
            buf,
            np.int32(total)
        )
        evt.wait()

        out_flat = np.empty_like(flat)
        cl.enqueue_copy(self.queue, out_flat, buf).wait()
        return out_flat.reshape(X_np.shape)

    def gpu_softmax_2d_inplace(self, X_np: np.ndarray) -> np.ndarray:
        """
        rowwise softmax on shape (B,T).
        """
        B,T = X_np.shape
        mf = cl.mem_flags
        buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=X_np)
        evt = self.program.softmax_2d(
            self.queue, (B,), None,
            buf,
            np.int32(B),
            np.int32(T)
        )
        evt.wait()

        out = np.empty((B,T), dtype=np.float32)
        cl.enqueue_copy(self.queue, out, buf).wait()
        return out


##############################################################################
# Optional test
##############################################################################
if __name__ == "__main__":
    print("=== Testing opencl_backend.py ===")

    backend = OpenCLBackend()
    print("[Test] Created OpenCLBackend")

    # 1) Test matmul
    A = np.random.randn(5,6).astype(np.float32)
    B = np.random.randn(6,4).astype(np.float32)
    C = backend.gpu_matmul(A, B)
    print("[Test] C shape = ", C.shape)  # (5,4)

    # 2) Test add_bias
    bias = np.random.randn(4).astype(np.float32)
    C_biased = backend.gpu_add_bias(C, bias)
    print("[Test] add_bias shape = ", C_biased.shape)

    # 3) ReLU
    C_relu = backend.gpu_relu_inplace(C_biased)
    print("[Test] ReLU shape = ", C_relu.shape)

    # 4) Softmax rowwise
    # shape => (B,T) => e.g. (3,5)
    X = np.random.randn(3,5).astype(np.float32)
    X_sm = backend.gpu_softmax_2d_inplace(X)
    print("[Test] softmax shape =", X_sm.shape)

    print("=== Done testing opencl_backend.py ===")
