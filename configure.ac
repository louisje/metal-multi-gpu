AC_INIT([metal-multi-gpu], [0.3.0], [none])
AM_INIT_AUTOMAKE([foreign subdir-objects])
AC_CONFIG_SRCDIR([src/cpu-benchmark.cpp])
AC_CONFIG_HEADERS([config.h])
AC_CONFIG_FILES([Makefile])

AC_PROG_CXX
AC_LANG([C++])

AC_MSG_CHECKING([whether the compiler supports AVX2/FMA intrinsics])
AC_COMPILE_IFELSE([
  AC_LANG_SOURCE([[
    #include <immintrin.h>
    __attribute__((target("avx2,fma")))
    static __m256 test_fma(__m256 a, __m256 b, __m256 c) {
      return _mm256_fmadd_ps(a, b, c);
    }
    int main() { return 0; }
  ]])
], [
  AC_MSG_RESULT([yes])
  AC_DEFINE([HAVE_AVX2_INTRINSICS], [1],
            [Compiler supports AVX2/FMA intrinsics])
], [
  AC_MSG_RESULT([no])
])

have_metal=no
METAL_CXXFLAGS=""
METAL_LIBS=""

AC_MSG_CHECKING([for Apple Metal support])
case "$(uname -s 2>/dev/null)" in
  Darwin)
    save_CXXFLAGS="$CXXFLAGS"
    save_LIBS="$LIBS"

    CXXFLAGS="$CXXFLAGS -x objective-c++ -fobjc-arc -fblocks"
    LIBS="$LIBS -framework Foundation -framework Metal"

    AC_LINK_IFELSE([
      AC_LANG_SOURCE([[
        #import <Foundation/Foundation.h>
        #import <Metal/Metal.h>

        int main() {
          @autoreleasepool {
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            return device ? 0 : 0;
          }
        }
      ]])
    ], [
      AC_MSG_RESULT([yes])
      have_metal=yes
      METAL_CXXFLAGS="-x objective-c++ -fobjc-arc -fblocks"
      METAL_LIBS="-framework Foundation -framework Metal"
      AC_DEFINE([HAVE_METAL], [1], [Apple Metal is available])
    ], [
      AC_MSG_RESULT([no])
    ])

    CXXFLAGS="$save_CXXFLAGS"
    LIBS="$save_LIBS"
    ;;
  *)
    AC_MSG_RESULT([no])
    ;;
esac

AM_CONDITIONAL([HAVE_METAL], [test "x$have_metal" = "xyes"])
AC_SUBST([METAL_CXXFLAGS])
AC_SUBST([METAL_LIBS])

AC_MSG_NOTICE([])
AC_MSG_NOTICE([Configuration summary:])
AC_MSG_NOTICE([  C++ compiler:       $CXX])
AC_MSG_NOTICE([  CPU benchmark:      yes])
AC_MSG_NOTICE([  Metal benchmarks:   $have_metal])

AC_OUTPUT
