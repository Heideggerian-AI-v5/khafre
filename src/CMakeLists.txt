cmake_minimum_required(VERSION 3.17)
project(khafre)

# Find the Python development files
set(Python3_FIND_VIRTUALENV "FIRST")
find_package(Python3 COMPONENTS Interpreter Development NumPy)

# Add the module to compile
Python3_add_library(contactcpp MODULE "contact.cpp")
target_include_directories(contactcpp
  PUBLIC
    ${Python3_NumPy_INCLUDE_DIRS}
)

# Install the module
install(TARGETS contactcpp
        EXCLUDE_FROM_ALL
        COMPONENT khafre_numerics
        DESTINATION ${PY_BUILD_CMAKE_MODULE_NAME})

# Install stubs to get autocomplete and type hints
install(FILES contactcpp.pyi
        EXCLUDE_FROM_ALL
        COMPONENT khafre_numerics
        DESTINATION ${PY_BUILD_CMAKE_MODULE_NAME})
