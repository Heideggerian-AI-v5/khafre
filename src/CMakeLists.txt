cmake_minimum_required(VERSION 3.17)
project(khafre)

# Find the Python development files
find_package(Python3 REQUIRED COMPONENTS Development.Module)

# Add the module to compile
Python3_add_library(contactcpp MODULE "contact.cpp")

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
