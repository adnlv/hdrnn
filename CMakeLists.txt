cmake_minimum_required(VERSION 3.28)

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
	cmake_policy(SET CMP0141 NEW)
	set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project(hdrnn)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

add_executable(hdrnn 
	"src/dataset.c" 
	"src/neunet.c"
	"src/main.c"
)

set(COMPILE_OPTIONS -Wall -Wextra -Wpedantic)
set(DEBUG_COMPILE_OPTIONS ${COMPILE_OPTIONS} -g3)
set(RELEASE_COMPILE_OPTIONS ${COMPILE_OPTIONS} -O3 -Werror)

target_compile_options(hdrnn
	PRIVATE
	$<$<AND:$<CONFIG:Debug>,$<OR:$<C_COMPILER_ID:GNU>,$<C_COMPILER_ID:Clang>>>:$<JOIN:${DEBUG_COMPILE_OPTIONS},;>>
	$<$<AND:$<CONFIG:Release>,$<OR:$<C_COMPILER_ID:GNU>,$<C_COMPILER_ID:Clang>>>:$<JOIN:${RELEASE_COMPILE_OPTIONS},;>>
)

if (UNIX)
	target_link_libraries(hdrnn PRIVATE m)

	# On Windows, you have to manually copy assets folder into the build one,
	# because it has a different concept of symbolic links.
	set(ASSETS_SOURCE_DIRECTORY "${CMAKE_SOURCE_DIR}/assets")
	set(ASSETS_LINK_DIRECTORY "${CMAKE_BINARY_DIR}/assets")

	add_custom_command(TARGET hdrnn POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E create_symlink ${ASSETS_SOURCE_DIRECTORY} ${ASSETS_LINK_DIRECTORY}
		COMMENT "Creating symbolic link to assets"
	)
endif()