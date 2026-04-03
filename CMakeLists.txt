cmake_minimum_required(VERSION 3.28)
project(hdrnn DESCRIPTION "Hand-written digit recognition neural network" LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

add_executable(hdrnn main.c mnist.c)

set(COMPILE_OPTIONS -Wall -Wextra -Wpedantic)
set(DEBUG_COMPILE_OPTIONS ${COMPILE_OPTIONS} -g3)
set(RELEASE_COMPILE_OPTIONS ${COMPILE_OPTIONS} -O3 -Werror)

target_compile_options(hdrnn
        PRIVATE
        $<$<AND:$<CONFIG:Debug>,$<OR:$<C_COMPILER_ID:GNU>,$<C_COMPILER_ID:Clang>>>:${DEBUG_COMPILE_OPTIONS}>
        $<$<AND:$<CONFIG:Debug>,$<OR:$<C_COMPILER_ID:GNU>,$<C_COMPILER_ID:Clang>>>:${RELEASE_COMPILE_OPTIONS}>
)
