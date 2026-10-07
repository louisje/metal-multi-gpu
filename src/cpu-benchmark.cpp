#include "config.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#if defined(HAVE_AVX2_INTRINSICS)
#include <immintrin.h>
#endif

namespace {

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

bool runtimeHasAvx2Fma()
{
#if defined(HAVE_AVX2_INTRINSICS) &&     (defined(__x86_64__) || defined(__i386__)) &&     (defined(__clang__) || defined(__GNUC__))
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") &&
           __builtin_cpu_supports("fma");
#else
    return false;
#endif
}

float scalarStep(float x, std::uint32_t rounds)
{
    for (std::uint32_t r = 0; r < rounds; ++r) {
        x = std::fma(x, 1.000001f, 0.000001f);
        x = std::fma(x, 0.999999f, 0.000002f);
        x = std::fma(x, 1.0000005f, -0.000001f);
        x = std::fma(x, 0.9999995f, 0.000003f);
    }

    return x;
}

double fmaGflops(
    std::size_t elements,
    std::uint32_t rounds,
    double milliseconds)
{
    const double operations =
        static_cast<double>(elements) *
        static_cast<double>(rounds) *
        8.0;

    return operations / (milliseconds / 1000.0) / 1.0e9;
}

Result runScalarFma(
    const std::vector<float> &input,
    std::vector<float> &output,
    std::uint32_t rounds)
{
    const auto start = std::chrono::steady_clock::now();

    for (std::size_t i = 0; i < input.size(); ++i) {
        output[i] = scalarStep(input[i], rounds);
    }

    const auto end = std::chrono::steady_clock::now();
    const double ms = elapsedMs(start, end);

    return {ms, fmaGflops(input.size(), rounds, ms)};
}

#if defined(HAVE_AVX2_INTRINSICS)
__attribute__((target("avx2,fma")))
void runFmaRangeAvx2(
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
        output[i] = scalarStep(input[i], rounds);
    }
}
#endif

Result runParallelFma(
    const std::vector<float> &input,
    std::vector<float> &output,
    std::uint32_t rounds,
    unsigned threadCount,
    bool useAvx2)
{
    threadCount = std::max(1u, threadCount);
    const auto start = std::chrono::steady_clock::now();

    std::vector<std::thread> workers;
    workers.reserve(threadCount);

    for (unsigned t = 0; t < threadCount; ++t) {
        const std::size_t begin = input.size() * t / threadCount;
        const std::size_t end = input.size() * (t + 1) / threadCount;

        workers.emplace_back([&, begin, end] {
#if defined(HAVE_AVX2_INTRINSICS)
            if (useAvx2) {
                runFmaRangeAvx2(
                    input.data(),
                    output.data(),
                    begin,
                    end,
                    rounds);
                return;
            }
#else
            (void)useAvx2;
#endif
            for (std::size_t i = begin; i < end; ++i) {
                output[i] = scalarStep(input[i], rounds);
            }
        });
    }

    for (std::thread &worker : workers) {
        worker.join();
    }

    const auto end = std::chrono::steady_clock::now();
    const double ms = elapsedMs(start, end);

    return {ms, fmaGflops(input.size(), rounds, ms)};
}

bool verify(
    const std::vector<float> &reference,
    const std::vector<float> &candidate,
    float tolerance = 1.0e-3f)
{
    if (reference.size() != candidate.size()) {
        return false;
    }

    for (std::size_t i = 0; i < reference.size(); ++i) {
        if (std::fabs(reference[i] - candidate[i]) > tolerance) {
            return false;
        }
    }

    return true;
}

void printResult(const std::string &name, const Result &result, bool ok = true)
{
    std::cout
        << std::left << std::setw(32)
        << (name + (ok ? " [OK]" : " [FAIL]"))
        << std::right << std::fixed << std::setprecision(3)
        << std::setw(10) << result.milliseconds << " ms"
        << std::setw(12) << result.gflops << " GFLOP/s"
        << "\n";
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

Result runScalarMatrix(
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
                sum = std::fma(a[row * n + k], b[k * n + col], sum);
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

#if defined(HAVE_AVX2_INTRINSICS)
__attribute__((target("avx2,fma")))
float dotAvx2(const float *a, const float *b, std::size_t n)
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

    float result = 0.0f;
    for (float lane : lanes) {
        result += lane;
    }

    for (; k < n; ++k) {
        result = std::fma(a[k], b[k], result);
    }

    return result;
}
#endif

Result runParallelMatrix(
    const std::vector<float> &a,
    const std::vector<float> &bTransposed,
    std::vector<float> &c,
    std::size_t n,
    unsigned threadCount,
    bool useAvx2)
{
    threadCount = std::max(1u, threadCount);
    threadCount = std::min<unsigned>(threadCount, static_cast<unsigned>(n));

    const auto start = std::chrono::steady_clock::now();

    std::vector<std::thread> workers;
    workers.reserve(threadCount);

    for (unsigned t = 0; t < threadCount; ++t) {
        const std::size_t rowBegin = n * t / threadCount;
        const std::size_t rowEnd = n * (t + 1) / threadCount;

        workers.emplace_back([&, rowBegin, rowEnd] {
            for (std::size_t row = rowBegin; row < rowEnd; ++row) {
                const float *aRow = a.data() + row * n;

                for (std::size_t col = 0; col < n; ++col) {
                    const float *bRow = bTransposed.data() + col * n;

#if defined(HAVE_AVX2_INTRINSICS)
                    if (useAvx2) {
                        c[row * n + col] = dotAvx2(aRow, bRow, n);
                        continue;
                    }
#else
                    (void)useAvx2;
#endif
                    float sum = 0.0f;
                    for (std::size_t k = 0; k < n; ++k) {
                        sum = std::fma(aRow[k], bRow[k], sum);
                    }
                    c[row * n + col] = sum;
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

int runFmaBenchmark(std::size_t elements, std::uint32_t rounds)
{
    const unsigned threads =
        std::max(1u, std::thread::hardware_concurrency());

    const bool avx2 = runtimeHasAvx2Fma();

    std::vector<float> input(elements);
    std::vector<float> reference(elements);
    std::vector<float> candidate(elements);

    for (std::size_t i = 0; i < elements; ++i) {
        input[i] = 0.5f + static_cast<float>(i % 1024) * 0.0001f;
    }

    std::cout
        << "CPU FMA benchmark\n"
        << "Threads: " << threads << "\n"
        << "Runtime AVX2+FMA: " << (avx2 ? "yes" : "no") << "\n\n";

    const Result scalar =
        runScalarFma(input, reference, rounds);

    printResult("CPU scalar", scalar);

    const Result parallel =
        runParallelFma(input, candidate, rounds, threads, false);

    printResult(
        "CPU scalar " + std::to_string(threads) + " threads",
        parallel,
        verify(reference, candidate));

    if (avx2) {
        const Result simd =
            runParallelFma(input, candidate, rounds, threads, true);

        printResult(
            "CPU AVX2+FMA " + std::to_string(threads) + " threads",
            simd,
            verify(reference, candidate));
    } else {
        std::cout << "CPU AVX2+FMA                  [SKIPPED]\n";
    }

    return 0;
}

int runMatrixBenchmark(std::size_t n)
{
    const unsigned threads =
        std::max(1u, std::thread::hardware_concurrency());

    const bool avx2 = runtimeHasAvx2Fma();

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

    std::cout
        << "CPU matrix benchmark\n"
        << "Matrix: " << n << " x " << n << "\n"
        << "Threads: " << threads << "\n"
        << "Runtime AVX2+FMA: " << (avx2 ? "yes" : "no") << "\n\n";

    const Result scalar =
        runScalarMatrix(a, b, reference, n);

    printResult("CPU scalar", scalar);

    const std::vector<float> bTransposed = transpose(b, n);

    const Result optimized =
        runParallelMatrix(
            a,
            bTransposed,
            candidate,
            n,
            threads,
            avx2);

    printResult(
        avx2
            ? "CPU AVX2+FMA " + std::to_string(threads) + " threads"
            : "CPU transposed " + std::to_string(threads) + " threads",
        optimized,
        verify(reference, candidate, 0.02f));

    return 0;
}

int selfTest()
{
    if (runFmaBenchmark(4096, 8) != 0) {
        return 1;
    }

    std::cout << "\n";

    if (runMatrixBenchmark(32) != 0) {
        return 1;
    }

    std::cout << "\nSelf-test passed.\n";
    return 0;
}

void printUsage(const char *program)
{
    std::cout
        << "Usage:\n"
        << "  " << program << " fma [elements] [rounds]\n"
        << "  " << program << " matrix [size]\n"
        << "  " << program << " --self-test\n";
}

} // namespace

int main(int argc, char **argv)
{
    if (argc == 1) {
        return runFmaBenchmark(1u << 20, 256);
    }

    const std::string mode = argv[1];

    if (mode == "--self-test") {
        return selfTest();
    }

    if (mode == "fma") {
        const std::size_t elements =
            argc > 2 ? std::stoull(argv[2]) : (1u << 20);

        const std::uint32_t rounds =
            argc > 3
                ? static_cast<std::uint32_t>(std::stoul(argv[3]))
                : 256;

        return runFmaBenchmark(elements, rounds);
    }

    if (mode == "matrix") {
        const std::size_t n =
            argc > 2 ? std::stoull(argv[2]) : 256;

        if (n == 0 || n > 4096) {
            std::cerr << "Matrix size must be between 1 and 4096.\n";
            return 1;
        }

        return runMatrixBenchmark(n);
    }

    printUsage(argv[0]);
    return 1;
}
