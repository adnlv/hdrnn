cmake_minimum_required(VERSION 3.28)
project(hdrnn DESCRIPTION "Hand-written digit recognition neural network" LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

add_executable(hdrnn
        main.c
        dataset.c)

set(COMPILE_OPTIONS -Wall -Wextra -Wpedantic)
set(DEBUG_COMPILE_OPTIONS ${COMPILE_OPTIONS} -g3)
set(RELEASE_COMPILE_OPTIONS ${COMPILE_OPTIONS} -O3 -Werror)

target_compile_options(hdrnn
        PRIVATE
        $<$<AND:$<CONFIG:Debug>,$<OR:$<C_COMPILER_ID:GNU>,$<C_COMPILER_ID:Clang>>>:$<JOIN:${DEBUG_COMPILE_OPTIONS},;>>
        $<$<AND:$<CONFIG:Release>,$<OR:$<C_COMPILER_ID:GNU>,$<C_COMPILER_ID:Clang>>>:$<JOIN:${RELEASE_COMPILE_OPTIONS},;>>
)

set(ASSETS_SOURCE_DIRECTORY "${CMAKE_SOURCE_DIR}/assets")
set(ASSETS_LINK_DIRECTORY "${CMAKE_BINARY_DIR}/assets")

add_custom_command(TARGET hdrnn POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E create_symlink ${ASSETS_SOURCE_DIRECTORY} ${ASSETS_LINK_DIRECTORY}
        COMMENT "Creating symbolic link to assets"
)
