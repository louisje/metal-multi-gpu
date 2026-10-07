#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <immintrin.h>
#include <iostream>
#include <limits>
#include <string>
#include <thread>
#include <vector>

namespace {

bool runtimeHasAvx2Fma()
{
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__clang__) || defined(__GNUC__))
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") &&
           __builtin_cpu_supports("fma");
#else
    return false;
#endif
}

constexpr std::size_t kDefaultMatrixSize = 256;
constexpr NSUInteger kTileSize = 16;
constexpr int kGpuRepeats = 5;

const char *kMetalSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

constant uint TILE = 16;

kernel void matmul_tiled(
    device const float *a [[buffer(0)]],
    device const float *b [[buffer(1)]],
    device float *c       [[buffer(2)]],
    constant uint &n      [[buffer(3)]],
    ushort2 tid           [[thread_position_in_threadgroup]],
    uint2 gid             [[thread_position_in_grid]])
{
    threadgroup float tileA[16][16];
    threadgroup float tileB[16][16];

    const uint row = gid.y;
    const uint col = gid.x;

    float sum = 0.0f;
    const uint tileCount = (n + TILE - 1) / TILE;

    for (uint tile = 0; tile < tileCount; ++tile) {
        const uint aCol = tile * TILE + tid.x;
        const uint bRow = tile * TILE + tid.y;

        tileA[tid.y][tid.x] =
            (row < n && aCol < n) ? a[row * n + aCol] : 0.0f;

        tileB[tid.y][tid.x] =
            (bRow < n && col < n) ? b[bRow * n + col] : 0.0f;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TILE; ++k) {
            sum = fma(tileA[tid.y][k], tileB[k][tid.x], sum);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < n && col < n) {
        c[row * n + col] = sum;
    }
}
)METAL";

struct MetalGpu {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> pipeline;
};

struct Result {
    double milliseconds;
    double gflops;
};

double elapsedMs(
    std::chrono::steady_clock::time_point start,
    std::chrono::steady_clock::time_point end)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

double matrixGflops(std::size_t n, double milliseconds)
{
    const double operations =
        2.0 *
        static_cast<double>(n) *
        static_cast<double>(n) *
        static_cast<double>(n);

    return operations / (milliseconds / 1000.0) / 1.0e9;
}

MetalGpu createGpu(id<MTLDevice> device)
{
    NSError *error = nil;

    NSString *source = [NSString stringWithUTF8String:kMetalSource];
    id<MTLLibrary> library =
        [device newLibraryWithSource:source options:nil error:&error];

    if (!library) {
        throw std::runtime_error(error.localizedDescription.UTF8String);
    }

    id<MTLFunction> function =
        [library newFunctionWithName:@"matmul_tiled"];

    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:function error:&error];

    if (!pipeline) {
        throw std::runtime_error(error.localizedDescription.UTF8String);
    }

    return {
        device,
        [device newCommandQueue],
        pipeline,
    };
}

Result benchmarkScalar(
    const std::vector<float> &a,
    const std::vector<float> &b,
    std::vector<float> &c,
    std::size_t n)
{
    const auto start = std::chrono::steady_clock::now();

    for (std::size_t row = 0; row < n; ++row) {
        for (std::size_t col = 0; col < n; ++col) {
            float sum = 0.0f;

            for (std::size_t k = 0; k < n; ++k) {
                sum = std::fma(
                    a[row * n + k],
                    b[k * n + col],
                    sum);
            }

            c[row * n + col] = sum;
        }
    }

    const auto end = std::chrono::steady_clock::now();
    const double ms = elapsedMs(start, end);

    return {ms, matrixGflops(n, ms)};
}

std::vector<float> transpose(
    const std::vector<float> &matrix,
    std::size_t n)
{
    std::vector<float> result(n * n);

    for (std::size_t row = 0; row < n; ++row) {
        for (std::size_t col = 0; col < n; ++col) {
            result[col * n + row] = matrix[row * n + col];
        }
    }

    return result;
}

__attribute__((target("avx2,fma")))
float dotProductAvx2(
    const float *a,
    const float *b,
    std::size_t n)
{
    __m256 sum = _mm256_setzero_ps();
    std::size_t k = 0;

    for (; k + 8 <= n; k += 8) {
        const __m256 va = _mm256_loadu_ps(a + k);
        const __m256 vb = _mm256_loadu_ps(b + k);
        sum = _mm256_fmadd_ps(va, vb, sum);
    }

    alignas(32) float lanes[8];
    _mm256_store_ps(lanes, sum);

    float result =
        lanes[0] + lanes[1] + lanes[2] + lanes[3] +
        lanes[4] + lanes[5] + lanes[6] + lanes[7];

    for (; k < n; ++k) {
        result = std::fma(a[k], b[k], result);
    }

    return result;
}

Result benchmarkAvx2Parallel(
    const std::vector<float> &a,
    const std::vector<float> &bTransposed,
    std::vector<float> &c,
    std::size_t n,
    unsigned threadCount)
{
    threadCount = std::max(1u, threadCount);
    threadCount = std::min<unsigned>(
        threadCount,
        static_cast<unsigned>(n));

    const auto start = std::chrono::steady_clock::now();

    std::vector<std::thread> workers;
    workers.reserve(threadCount);

    for (unsigned threadIndex = 0; threadIndex < threadCount; ++threadIndex) {
        const std::size_t rowBegin =
            n * threadIndex / threadCount;
        const std::size_t rowEnd =
            n * (threadIndex + 1) / threadCount;

        workers.emplace_back([&, rowBegin, rowEnd] {
            for (std::size_t row = rowBegin; row < rowEnd; ++row) {
                const float *aRow = a.data() + row * n;

                for (std::size_t col = 0; col < n; ++col) {
                    const float *bRow =
                        bTransposed.data() + col * n;

                    c[row * n + col] =
                        dotProductAvx2(aRow, bRow, n);
                }
            }
        });
    }

    for (std::thread &worker : workers) {
        worker.join();
    }

    const auto end = std::chrono::steady_clock::now();
    const double ms = elapsedMs(start, end);

    return {ms, matrixGflops(n, ms)};
}

Result benchmarkGpu(
    const MetalGpu &gpu,
    const std::vector<float> &a,
    const std::vector<float> &b,
    std::vector<float> &c,
    std::size_t n)
{
    const std::size_t bytes = n * n * sizeof(float);
    const uint32_t n32 = static_cast<uint32_t>(n);

    id<MTLBuffer> aBuffer =
        [gpu.device newBufferWithBytes:a.data()
                               length:bytes
                              options:MTLResourceStorageModeShared];

    id<MTLBuffer> bBuffer =
        [gpu.device newBufferWithBytes:b.data()
                               length:bytes
                              options:MTLResourceStorageModeShared];

    id<MTLBuffer> cBuffer =
        [gpu.device newBufferWithLength:bytes
                                options:MTLResourceStorageModeShared];

    id<MTLBuffer> nBuffer =
        [gpu.device newBufferWithBytes:&n32
                               length:sizeof(n32)
                              options:MTLResourceStorageModeShared];

    const MTLSize threadsPerThreadgroup =
        MTLSizeMake(kTileSize, kTileSize, 1);

    const MTLSize grid =
        MTLSizeMake(n, n, 1);

    double bestMs = std::numeric_limits<double>::infinity();

    for (int repeat = 0; repeat < kGpuRepeats; ++repeat) {
        id<MTLCommandBuffer> commandBuffer =
            [gpu.queue commandBuffer];

        id<MTLComputeCommandEncoder> encoder =
            [commandBuffer computeCommandEncoder];

        [encoder setComputePipelineState:gpu.pipeline];
        [encoder setBuffer:aBuffer offset:0 atIndex:0];
        [encoder setBuffer:bBuffer offset:0 atIndex:1];
        [encoder setBuffer:cBuffer offset:0 atIndex:2];
        [encoder setBuffer:nBuffer offset:0 atIndex:3];

        [encoder dispatchThreads:grid
           threadsPerThreadgroup:threadsPerThreadgroup];

        [encoder endEncoding];

        const auto start = std::chrono::steady_clock::now();
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        const auto end = std::chrono::steady_clock::now();

        bestMs = std::min(bestMs, elapsedMs(start, end));
    }

    std::memcpy(
        c.data(),
        cBuffer.contents,
        bytes);

    return {bestMs, matrixGflops(n, bestMs)};
}

bool verify(
    const std::vector<float> &reference,
    const std::vector<float> &candidate)
{
    constexpr float kAbsoluteTolerance = 0.02f;

    for (std::size_t i = 0; i < reference.size(); ++i) {
        const float difference =
            std::fabs(reference[i] - candidate[i]);

        if (difference > kAbsoluteTolerance) {
            std::cerr
                << "Mismatch at " << i
                << ": expected " << reference[i]
                << ", got " << candidate[i]
                << ", difference " << difference
                << "\n";
            return false;
        }
    }

    return true;
}

void printResult(
    const std::string &name,
    const Result &result,
    bool ok = true)
{
    std::cout
        << std::left << std::setw(34)
        << (name + (ok ? " [OK]" : " [FAIL]"))
        << std::right << std::fixed << std::setprecision(3)
        << std::setw(10) << result.milliseconds << " ms"
        << std::setw(12) << result.gflops << " GFLOP/s"
        << "\n";
}

} // namespace

int main(int argc, char **argv)
{
    @autoreleasepool {
        const std::size_t n =
            argc > 1 ? std::stoull(argv[1]) : kDefaultMatrixSize;

        if (n == 0 || n > 2048) {
            std::cerr << "Matrix size must be between 1 and 2048.\n";
            return 1;
        }

        const unsigned cpuThreads =
            std::max(1u, std::thread::hardware_concurrency());

        NSArray<id<MTLDevice>> *devices = MTLCopyAllDevices();

        std::cout
            << "Matrix multiplication benchmark\n"
            << "Matrix size: " << n << " x " << n << "\n"
            << "CPU threads: " << cpuThreads << "\n"
            << "Metal devices: " << devices.count << "\n";

        std::vector<MetalGpu> gpus;
        for (id<MTLDevice> device in devices) {
            std::cout
                << "  - " << device.name.UTF8String
                << "\n";

            gpus.push_back(createGpu(device));
        }

        if (gpus.empty()) {
            std::cerr << "No Metal GPU found.\n";
            return 1;
        }

        std::vector<float> a(n * n);
        std::vector<float> b(n * n);
        std::vector<float> reference(n * n);
        std::vector<float> candidate(n * n);

        for (std::size_t row = 0; row < n; ++row) {
            for (std::size_t col = 0; col < n; ++col) {
                a[row * n + col] =
                    0.001f * static_cast<float>((row + col) % 97);

                b[row * n + col] =
                    0.001f * static_cast<float>((row * 3 + col * 5) % 89);
            }
        }

        std::cout << "\n";

        const Result scalarResult =
            benchmarkScalar(a, b, reference, n);

        printResult("CPU scalar", scalarResult);

        if (runtimeHasAvx2Fma()) {
            const std::vector<float> bTransposed =
                transpose(b, n);

            const Result avx2Result =
                benchmarkAvx2Parallel(
                    a,
                    bTransposed,
                    candidate,
                    n,
                    cpuThreads);

            printResult(
                "CPU AVX2+FMA " +
                    std::to_string(cpuThreads) +
                    " threads",
                avx2Result,
                verify(reference, candidate));
        } else {
            std::cout << "CPU AVX2+FMA: [SKIPPED - unsupported CPU]\n";
        }

        for (const MetalGpu &gpu : gpus) {
            const Result gpuResult =
                benchmarkGpu(
                    gpu,
                    a,
                    b,
                    candidate,
                    n);

            printResult(
                std::string(gpu.device.name.UTF8String),
                gpuResult,
                verify(reference, candidate));
        }
    }

    return 0;
}
