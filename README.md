# metal-multi-gpu

Portable CPU and Apple Metal compute benchmarks for Intel/x86 systems.

The build is configured with Autoconf. CPU benchmarks are always built.
Metal benchmarks are enabled automatically only when Apple Metal and
Foundation are available.

## Build

For a normal source checkout or release tarball:

```sh
./configure
make
make check
```

The configure summary shows which optional backends are available:

```text
Configuration summary:
  C++ compiler:       g++
  CPU benchmark:      yes
  Metal benchmarks:   yes
```

On Linux or another non-Metal platform, the CPU benchmark still builds
while the Metal targets are omitted.

## Developer bootstrap

If `configure.ac` changes, regenerate the generated files with:

```sh
./bootstrap
```

The developer bootstrap uses Autoconf and Automake. It generates `configure`, `config.h.in`, and `Makefile.in`. These generated files can be shipped so users do not need Autotools installed just to build the project.

## Portable CPU benchmark

`cpu-benchmark` is plain C++17 and has no Metal dependency.

FMA-heavy workload:

```sh
./cpu-benchmark fma
./cpu-benchmark fma 1048576 256
```

Matrix multiplication:

```sh
./cpu-benchmark matrix
./cpu-benchmark matrix 512
```

The configure step checks whether the compiler can build AVX2/FMA
intrinsics. At runtime the program separately checks whether the actual
CPU supports AVX2 and FMA. Unsupported CPUs automatically skip the SIMD
path instead of executing unsupported instructions.

## Metal FMA benchmark

On macOS systems where Metal is available:

```sh
./metal-multi-gpu
./metal-multi-gpu 1048576 256
```

It compares scalar CPU, multi-threaded CPU, AVX2/FMA CPU, and each Metal
GPU exposed by macOS. GPU buffers are allocated outside the timed region
and the GPU measurement is repeated five times.

## Metal matrix benchmark

```sh
./matrix-benchmark
./matrix-benchmark 512
```

The GPU implementation uses a tiled 16x16 Metal compute kernel. The CPU
side uses a scalar reference implementation plus AVX2/FMA and multiple
threads when the running CPU supports them.

## Source layout

```text
configure.ac
Makefile.am
Makefile.in
bootstrap
src/
  cpu-benchmark.cpp
  fma-metal.mm
  matrix-metal.mm
```

`src/cpu-benchmark.cpp` is the portable path. The Objective-C++ files
are compiled only when configure detects Apple Metal.

## Portability model

There are two distinct checks for SIMD:

1. Configure-time: can this compiler build AVX2/FMA intrinsics?
2. Runtime: does this particular CPU actually support AVX2 and FMA?

Metal is also configure-time optional. This lets the same source tree be
used on newer Intel Macs, older Intel Macs, and non-macOS x86 systems
without requiring Metal everywhere.
