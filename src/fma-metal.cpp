#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include <iomanip>
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

constexpr std::size_t kDefaultElements = 1u << 20;  // 1,048,576 floats
constexpr std::uint32_t kDefaultRounds = 256;
constexpr int kGpuRepeats = 5;

// Each round performs four FMAs.
// One FMA is conventionally counted as two floating-point operations.
constexpr double kFlopsPerRound = 8.0;

const char *kMetalSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

kernel void compute_heavy(
    device const float *input  [[buffer(0)]],
    device float *output       [[buffer(1)]],
    constant uint &rounds      [[buffer(2)]],
    uint id                    [[thread_position_in_grid]])
{
    float x = input[id];

    for (uint r = 0; r < rounds; ++r) {
        x = fma(x, 1.000001f, 0.000001f);
        x = fma(x, 0.999999f, 0.000002f);
        x = fma(x, 1.0000005f, -0.000001f);
        x = fma(x, 0.9999995f, 0.000003f);
    }

    output[id] = x;
}
)METAL";

struct MetalGpu {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> pipeline;
};

struct BenchmarkResult {
    double milliseconds;
    double gflops;
};

double elapsedMilliseconds(
    std::chrono::steady_clock::time_point start,
    std::chrono::steady_clock::time_point end)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

double calculateGflops(std::size_t elements, std::uint32_t rounds, double milliseconds)
{
    const double totalFlops =
        static_cast<double>(elements) *
        static_cast<double>(rounds) *
        kFlopsPerRound;

    return totalFlops / (milliseconds / 1000.0) / 1.0e9;
}

float computeReferenceValue(float x, std::uint32_t rounds)
{
    for (std::uint32_t r = 0; r < rounds; ++r) {
        x = std::fma(x, 1.000001f, 0.000001f);
        x = std::fma(x, 0.999999f, 0.000002f);
        x = std::fma(x, 1.0000005f, -0.000001f);
        x = std::fma(x, 0.9999995f, 0.000003f);
    }

    return x;
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
        [library newFunctionWithName:@"compute_heavy"];

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

__attribute__((target("avx2,fma")))
void computeRangeSimd(
    const float *input,
    float *output,
    std::size_t begin,
    std::size_t end,
    std::uint32_t rounds)
{
    const __m256 a1 = _mm256_set1_ps(1.000001f);
    const __m256 b1 = _mm256_set1_ps(0.000001f);
    const __m256 a2 = _mm256_set1_ps(0.999999f);
    const __m256 b2 = _mm256_set1_ps(0.000002f);
    const __m256 a3 = _mm256_set1_ps(1.0000005f);
    const __m256 b3 = _mm256_set1_ps(-0.000001f);
    const __m256 a4 = _mm256_set1_ps(0.9999995f);
    const __m256 b4 = _mm256_set1_ps(0.000003f);

    std::size_t i = begin;

    for (; i + 8 <= end; i += 8) {
        __m256 x = _mm256_loadu_ps(input + i);

        for (std::uint32_t r = 0; r < rounds; ++r) {
            x = _mm256_fmadd_ps(x, a1, b1);
            x = _mm256_fmadd_ps(x, a2, b2);
            x = _mm256_fmadd_ps(x, a3, b3);
            x = _mm256_fmadd_ps(x, a4, b4);
        }

        _mm256_storeu_ps(output + i, x);
    }

    for (; i < end; ++i) {
        output[i] = computeReferenceValue(input[i], rounds);
    }
}

BenchmarkResult benchmarkCpuSimd(
    const std::vector<float> &input,
    std::vector<float> &output,
    std::uint32_t rounds)
{
    const auto start = std::chrono::steady_clock::now();

    computeRangeSimd(
        input.data(),
        output.data(),
        0,
        input.size(),
        rounds);

    const auto end = std::chrono::steady_clock::now();
    const double ms = elapsedMilliseconds(start, end);

    return {ms, calculateGflops(input.size(), rounds, ms)};
}

BenchmarkResult benchmarkCpuSimdParallel(
    const std::vector<float> &input,
    std::vector<float> &output,
    std::uint32_t rounds,
    unsigned threadCount)
{
    threadCount = std::max(1u, threadCount);
    threadCount = std::min<unsigned>(
        threadCount,
        static_cast<unsigned>(input.size()));

    const auto start = std::chrono::steady_clock::now();

    std::vector<std::thread> workers;
    workers.reserve(threadCount);

    for (unsigned threadIndex = 0; threadIndex < threadCount; ++threadIndex) {
        const std::size_t begin =
            input.size() * threadIndex / threadCount;
        const std::size_t end =
            input.size() * (threadIndex + 1) / threadCount;

        workers.emplace_back([&, begin, end] {
            computeRangeSimd(
                input.data(),
                output.data(),
                begin,
                end,
                rounds);
        });
    }

    for (std::thread &worker : workers) {
        worker.join();
    }

    const auto end = std::chrono::steady_clock::now();
    const double ms = elapsedMilliseconds(start, end);

    return {ms, calculateGflops(input.size(), rounds, ms)};
}

BenchmarkResult benchmarkCpuParallel(
    const std::vector<float> &input,
    std::vector<float> &output,
    std::uint32_t rounds,
    unsigned threadCount)
{
    threadCount = std::max(1u, threadCount);
    threadCount = std::min<unsigned>(
        threadCount,
        static_cast<unsigned>(input.size()));

    const auto start = std::chrono::steady_clock::now();

    std::vector<std::thread> workers;
    workers.reserve(threadCount);

    for (unsigned threadIndex = 0; threadIndex < threadCount; ++threadIndex) {
        const std::size_t begin =
            input.size() * threadIndex / threadCount;
        const std::size_t end =
            input.size() * (threadIndex + 1) / threadCount;

        workers.emplace_back([&, begin, end] {
            for (std::size_t i = begin; i < end; ++i) {
                output[i] = computeReferenceValue(input[i], rounds);
            }
        });
    }

    for (std::thread &worker : workers) {
        worker.join();
    }

    const auto end = std::chrono::steady_clock::now();
    const double ms = elapsedMilliseconds(start, end);

    return {ms, calculateGflops(input.size(), rounds, ms)};
}

BenchmarkResult benchmarkCpu(
    const std::vector<float> &input,
    std::vector<float> &output,
    std::uint32_t rounds)
{
    const auto start = std::chrono::steady_clock::now();

    for (std::size_t i = 0; i < input.size(); ++i) {
        output[i] = computeReferenceValue(input[i], rounds);
    }

    const auto end = std::chrono::steady_clock::now();
    const double ms = elapsedMilliseconds(start, end);

    return {ms, calculateGflops(input.size(), rounds, ms)};
}

BenchmarkResult benchmarkGpu(
    const MetalGpu &gpu,
    const std::vector<float> &input,
    std::vector<float> &output,
    std::uint32_t rounds)
{
    const std::size_t bytes = input.size() * sizeof(float);

    // Allocate and upload once. This setup is intentionally outside the timer.
    id<MTLBuffer> inputBuffer =
        [gpu.device newBufferWithBytes:input.data()
                               length:bytes
                              options:MTLResourceStorageModeShared];

    id<MTLBuffer> outputBuffer =
        [gpu.device newBufferWithLength:bytes
                                options:MTLResourceStorageModeShared];

    id<MTLBuffer> roundsBuffer =
        [gpu.device newBufferWithBytes:&rounds
                               length:sizeof(rounds)
                              options:MTLResourceStorageModeShared];

    const NSUInteger threadWidth =
        std::min<NSUInteger>(256, gpu.pipeline.maxTotalThreadsPerThreadgroup);

    const MTLSize gridSize = MTLSizeMake(input.size(), 1, 1);
    const MTLSize threadgroupSize = MTLSizeMake(threadWidth, 1, 1);

    double bestMs = std::numeric_limits<double>::infinity();

    for (int repeat = 0; repeat < kGpuRepeats; ++repeat) {
        id<MTLCommandBuffer> commandBuffer = [gpu.queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder =
            [commandBuffer computeCommandEncoder];

        [encoder setComputePipelineState:gpu.pipeline];
        [encoder setBuffer:inputBuffer offset:0 atIndex:0];
        [encoder setBuffer:outputBuffer offset:0 atIndex:1];
        [encoder setBuffer:roundsBuffer offset:0 atIndex:2];

        [encoder dispatchThreads:gridSize
           threadsPerThreadgroup:threadgroupSize];

        [encoder endEncoding];

        const auto start = std::chrono::steady_clock::now();
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        const auto end = std::chrono::steady_clock::now();

        bestMs = std::min(bestMs, elapsedMilliseconds(start, end));
    }

    // Read back once, after the timed runs.
    std::memcpy(output.data(), outputBuffer.contents, bytes);

    return {
        bestMs,
        calculateGflops(input.size(), rounds, bestMs),
    };
}

bool verifyOutput(
    const std::vector<float> &reference,
    const std::vector<float> &candidate)
{
    constexpr float kTolerance = 1.0e-3f;

    for (std::size_t i = 0; i < reference.size(); ++i) {
        if (std::fabs(reference[i] - candidate[i]) > kTolerance) {
            std::cerr
                << "Mismatch at index " << i
                << ": expected " << reference[i]
                << ", got " << candidate[i]
                << "\n";
            return false;
        }
    }

    return true;
}

void printResult(const std::string &name, const BenchmarkResult &result)
{
    std::cout
        << std::left << std::setw(28) << name
        << std::right << std::fixed << std::setprecision(3)
        << std::setw(10) << result.milliseconds << " ms"
        << std::setw(12) << result.gflops << " GFLOP/s"
        << "\n";
}

} // namespace

int main(int argc, char **argv)
{
    @autoreleasepool {
        const std::size_t elementCount =
            argc > 1 ? std::stoull(argv[1]) : kDefaultElements;

        const std::uint32_t rounds =
            argc > 2 ? static_cast<std::uint32_t>(std::stoul(argv[2]))
                     : kDefaultRounds;

        NSArray<id<MTLDevice>> *devices = MTLCopyAllDevices();

        std::cout << "Metal devices: " << devices.count << "\n";

        std::vector<MetalGpu> gpus;
        for (id<MTLDevice> device in devices) {
            std::cout
                << "  - " << device.name.UTF8String
                << (device.lowPower ? " [Integrated/low-power]" : " [Discrete]")
                << "\n";

            gpus.push_back(createGpu(device));
        }

        if (gpus.empty()) {
            std::cerr << "No Metal GPU found.\n";
            return 1;
        }

        std::cout
            << "\nElements: " << elementCount
            << "\nRounds per element: " << rounds
            << "\nGPU runs: " << kGpuRepeats
            << " (best result shown)"
            << "\n\n";

        std::vector<float> input(elementCount);
        std::vector<float> cpuOutput(elementCount);
        std::vector<float> gpuOutput(elementCount);

        for (std::size_t i = 0; i < elementCount; ++i) {
            input[i] = 0.5f + static_cast<float>(i % 1024) * 0.0001f;
        }

        const unsigned cpuThreads =
            std::max(1u, std::thread::hardware_concurrency());

        const BenchmarkResult cpuSingleResult =
            benchmarkCpu(input, cpuOutput, rounds);

        printResult("CPU single-thread", cpuSingleResult);

        std::vector<float> cpuParallelOutput(elementCount);
        const BenchmarkResult cpuParallelResult =
            benchmarkCpuParallel(
                input,
                cpuParallelOutput,
                rounds,
                cpuThreads);

        const bool cpuParallelOk =
            verifyOutput(cpuOutput, cpuParallelOutput);

        printResult(
            "CPU " + std::to_string(cpuThreads) + " threads" +
                (cpuParallelOk ? " [OK]" : " [FAIL]"),
            cpuParallelResult);

        if (runtimeHasAvx2Fma()) {
            std::vector<float> cpuSimdOutput(elementCount);
            const BenchmarkResult cpuSimdResult =
                benchmarkCpuSimd(input, cpuSimdOutput, rounds);

            const bool cpuSimdOk =
                verifyOutput(cpuOutput, cpuSimdOutput);

            printResult(
                std::string("CPU AVX2+FMA") +
                    (cpuSimdOk ? " [OK]" : " [FAIL]"),
                cpuSimdResult);

            std::vector<float> cpuSimdParallelOutput(elementCount);
            const BenchmarkResult cpuSimdParallelResult =
                benchmarkCpuSimdParallel(
                    input,
                    cpuSimdParallelOutput,
                    rounds,
                    cpuThreads);

            const bool cpuSimdParallelOk =
                verifyOutput(cpuOutput, cpuSimdParallelOutput);

            printResult(
                "CPU AVX2+FMA " + std::to_string(cpuThreads) + " threads" +
                    (cpuSimdParallelOk ? " [OK]" : " [FAIL]"),
                cpuSimdParallelResult);
        } else {
            std::cout << "CPU AVX2+FMA: [SKIPPED - unsupported CPU]\n";
        }

        for (const MetalGpu &gpu : gpus) {
            const BenchmarkResult gpuResult =
                benchmarkGpu(gpu, input, gpuOutput, rounds);

            const bool ok = verifyOutput(cpuOutput, gpuOutput);

            printResult(
                std::string(gpu.device.name.UTF8String) +
                    (ok ? " [OK]" : " [FAIL]"),
                gpuResult);
        }
    }

    return 0;
}
