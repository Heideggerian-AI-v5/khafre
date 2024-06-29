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
    ${Python3_INCLUDE_DIRS}
)

if(PY_BUILD_CMAKE_MODULE_NAME)
  set(INSTALL_PATH ${PY_BUILD_CMAKE_MODULE_NAME})
else()
  include(GNUInstallDirs)
  set(INSTALL_PATH ${CMAKE_INSTALL_LIBDIR})
endif()

# Install the module
install(TARGETS contactcpp
        EXCLUDE_FROM_ALL
        COMPONENT khafre_numerics
        LIBRARY DESTINATION ${INSTALL_PATH})

# Install stubs to get autocomplete and type hints
install(FILES contactcpp.pyi
        EXCLUDE_FROM_ALL
        COMPONENT khafre_numerics
        DESTINATION ${INSTALL_PATH})
