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

set(WINDOWS_COMPILE_OPTIONS
	$<$<CONFIG:Debug>:/W4 /ZI> 
	$<$<CONFIG:Release>:/W4 /O2>
)

set(UNIX_COMPILE_OPTIONS 
	$<$<CONFIG:Debug>:-Wall -Wextra -Wpedantic -g3> 
	$<$<CONFIG:Release>:-Wall -Wextra -Wpedantic -Werror -O3>
)

set(HDRNN_COMPILE_OPTIONS 
	$<$<C_COMPILER_ID:MSVC>:${WINDOWS_COMPILE_OPTIONS}>
	$<$<OR:$<C_COMPILER_ID:GNU>,$<C_COMPILER_ID:Clang>>:${UNIX_COMPILE_OPTIONS}>
)

target_compile_options(hdrnn PRIVATE $<JOIN:${HDRNN_COMPILE_OPTIONS},;>)

if (UNIX)
	# On Windows the math library is linked automatically.
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