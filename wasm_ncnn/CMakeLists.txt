cmake_minimum_required(VERSION 3.10)
project(card-browser-demo)

set(CMAKE_BUILD_TYPE Release)
set(BUILD_SHARED_LIBS OFF)

include_directories(${CMAKE_SOURCE_DIR}/include)

# Prebuilt ncnn-wasm (simd variant)
set(ncnn_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ncnn/lib/cmake/ncnn")
find_package(ncnn REQUIRED)

# Emscripten compilation flags
set(EM_FLAGS "-O3 -msimd128 -s WASM=1 -s FORCE_FILESYSTEM=1 -s INITIAL_MEMORY=256MB -s EXIT_RUNTIME=1")

set(EXPORTED_RUNTIME_METHODS "[\"_free\",\"ccall\",\"cwrap\"]")
set(EXPORTED_FUNCTIONS "[\"_models_init\",\"_process_frame\",\"_malloc\"]")

# Bundle the assets/ folder (param/bin files for YOLO + 3 templates + 2 symbols + paddle + dict)
set(PRELOAD "--preload-file ${CMAKE_CURRENT_SOURCE_DIR}/assets@assets")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${EM_FLAGS} -s EXPORTED_FUNCTIONS=${EXPORTED_FUNCTIONS} -s EXPORTED_RUNTIME_METHODS=${EXPORTED_RUNTIME_METHODS} -s EMBIND=1 ${PRELOAD}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${EM_FLAGS} -s EXPORTED_FUNCTIONS=${EXPORTED_FUNCTIONS} -s EXPORTED_RUNTIME_METHODS=${EXPORTED_RUNTIME_METHODS} ${PRELOAD}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${EM_FLAGS} -s EXPORTED_FUNCTIONS=${EXPORTED_FUNCTIONS} -s EXPORTED_RUNTIME_METHODS=${EXPORTED_RUNTIME_METHODS} -s ALLOW_MEMORY_GROWTH=1 -s ENVIRONMENT=web -s USE_SDL=0 -s INITIAL_MEMORY=128MB ${PRELOAD} -lembind")

add_executable(cards src/main.cpp)
target_link_libraries(cards ncnn)
