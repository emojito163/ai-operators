cmake_minimum_required(VERSION 3.15)

set_source_files_properties(${CMAKE_SOURCE_DIR}/../src/vec_kernels.hu PROPERTIES LANGUAGE CXX)

project(tests LANGUAGES CXX)
set(CMAKE_CXX_COMPILER /usr/local/houmo/bin/clang++)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_BUILD_TYPE Debug)
enable_testing()

set(CMAKE_EXE_LINKER_FLAGS_INIT "-fuse-ld=lld -flto=thin")
set(CMAKE_MODULE_LINKER_FLAGS_INIT "-fuse-ld=lld -flto=thin")
set(CMAKE_SHARED_LINKER_FLAGS_INIT "-fuse-ld=lld -flto=thin")

file(GLOB_RECURSE TEST_SRCS ${CMAKE_SOURCE_DIR}/*.cpp)

add_compile_options(-x hdpl -fno-rtti -fno-exceptions -fno-unwind-tables -fno-asynchronous-unwind-tables -fno-ident)
include_directories(${CMAKE_SOURCE_DIR}/../include /usr/local/houmo/include)
link_directories(/usr/local/houmo/lib)

foreach (a_test IN LISTS TEST_SRCS)
    get_filename_component(filename ${a_test} NAME_WE)
    add_executable(${filename} ${a_test})
    target_link_libraries(${filename} PRIVATE hdplrt idnnl)
    add_test(NAME ${filename} COMMAND ${CMAKE_BINARY_DIR}/${filename})
endforeach()

