noinst_PROGRAMS = cpu-benchmark

cpu_benchmark_SOURCES = src/cpu-benchmark.cpp
cpu_benchmark_CPPFLAGS = -I$(top_builddir)

check-local: cpu-benchmark
	./cpu-benchmark --self-test

if HAVE_METAL
noinst_PROGRAMS += metal-multi-gpu matrix-benchmark

metal_multi_gpu_SOURCES = src/fma-metal.cpp
metal_multi_gpu_CPPFLAGS = -I$(top_builddir)
metal_multi_gpu_CXXFLAGS = $(AM_CXXFLAGS) $(METAL_CXXFLAGS)
metal_multi_gpu_LDFLAGS = -x none
metal_multi_gpu_LDADD = $(METAL_LIBS)

matrix_benchmark_SOURCES = src/matrix-metal.cpp
matrix_benchmark_CPPFLAGS = -I$(top_builddir)
matrix_benchmark_CXXFLAGS = $(AM_CXXFLAGS) $(METAL_CXXFLAGS)
matrix_benchmark_LDFLAGS = -x none
matrix_benchmark_LDADD = $(METAL_LIBS)
endif

EXTRA_DIST = README.md bootstrap
