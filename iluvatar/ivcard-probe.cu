#include <cstdio>
__global__ void vadd(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
int main() {
    const int n = 1024;
    float *a, *b, *c, ha[1024], hb[1024], hc[1024];
    cudaMalloc(&a, n*4); cudaMalloc(&b, n*4); cudaMalloc(&c, n*4);
    for (int i = 0; i < n; ++i) { ha[i] = i; hb[i] = 2*i; }
    cudaMemcpy(a, ha, n*4, cudaMemcpyHostToDevice);
    cudaMemcpy(b, hb, n*4, cudaMemcpyHostToDevice);
    vadd<<<(n+255)/256, 256>>>(a, b, c, n);
    cudaError_t e = cudaDeviceSynchronize();
    printf("sync: %s\n", cudaGetErrorString(e));
    cudaMemcpy(hc, c, n*4, cudaMemcpyDeviceToHost);
    int bad = 0;
    for (int i = 0; i < n; ++i) if (hc[i] != 3.0f*i) ++bad;
    printf("bad=%d  %s\n", bad, bad ? "FAIL" : "PASS");
    return bad != 0;
}
