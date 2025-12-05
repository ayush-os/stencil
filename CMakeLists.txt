cmake_minimum_required(VERSION 3.10)
project(stencil C CXX CUDA)

find_package(CUDA REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES "80" CACHE STRING "Target CUDA architectures")

set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)

add_executable(stencil
    src/host_main.cu
    src/kernels/kernel_baseline.cu
)

target_include_directories(stencil PRIVATE ${CUDA_INCLUDE_DIRS})

target_compile_options(stencil PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-std=c++14>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

target_link_libraries(stencil PRIVATE
    ${CUDA_LIBRARIES}
)

set_target_properties(stencil PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)