# cmake/FindNCCL.cmake

# NVIDIA does not distribute CMake files with NCCl, therefore use this file to find it instead.

find_path(NCCL_INCLUDE_DIR
    NAMES nccl.h
    HINTS ${NCCL_ROOT} $ENV{NCCL_ROOT} $ENV{CUDA_HOME} /usr/local/cuda
    PATH_SUFFIXES include
)

find_library(NCCL_LIBRARY
    NAMES nccl
    HINTS ${NCCL_ROOT} $ENV{NCCL_ROOT} $ENV{CUDA_HOME} /usr/local/cuda
    PATH_SUFFIXES lib lib64
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NCCL
    DEFAULT_MSG
    NCCL_LIBRARY NCCL_INCLUDE_DIR
)

if(NCCL_FOUND)
    set(NCCL_LIBRARIES ${NCCL_LIBRARY})
    set(NCCL_INCLUDE_DIRS ${NCCL_INCLUDE_DIR})

    if(NOT TARGET NCCL::NCCL)
        if(WIN32)
            # On Windows nccl.lib is an import library; the runtime DLL lives in bin/.
            # CMake requires both IMPORTED_IMPLIB (used by the linker) and
            # IMPORTED_LOCATION (the DLL) to correctly link a CUDA shared library.
            find_file(NCCL_DLL
                NAMES nccl.dll
                HINTS ${NCCL_ROOT} $ENV{NCCL_ROOT} $ENV{CUDA_HOME}
                PATH_SUFFIXES bin
            )
            add_library(NCCL::NCCL SHARED IMPORTED)
            set_target_properties(NCCL::NCCL PROPERTIES
                IMPORTED_IMPLIB               "${NCCL_LIBRARY}"
                IMPORTED_LOCATION             "${NCCL_DLL}"
                INTERFACE_INCLUDE_DIRECTORIES "${NCCL_INCLUDE_DIR}"
            )
        else()
            add_library(NCCL::NCCL UNKNOWN IMPORTED)
            set_target_properties(NCCL::NCCL PROPERTIES
                IMPORTED_LOCATION             "${NCCL_LIBRARY}"
                INTERFACE_INCLUDE_DIRECTORIES "${NCCL_INCLUDE_DIR}"
            )
        endif()
    endif()
endif()

mark_as_advanced(NCCL_INCLUDE_DIR NCCL_LIBRARY NCCL_DLL)
