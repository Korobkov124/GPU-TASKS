# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/content/GPU-TASKS/build/_deps/benchmark-src")
  file(MAKE_DIRECTORY "/content/GPU-TASKS/build/_deps/benchmark-src")
endif()
file(MAKE_DIRECTORY
  "/content/GPU-TASKS/build/_deps/benchmark-build"
  "/content/GPU-TASKS/build/_deps/benchmark-subbuild/benchmark-populate-prefix"
  "/content/GPU-TASKS/build/_deps/benchmark-subbuild/benchmark-populate-prefix/tmp"
  "/content/GPU-TASKS/build/_deps/benchmark-subbuild/benchmark-populate-prefix/src/benchmark-populate-stamp"
  "/content/GPU-TASKS/build/_deps/benchmark-subbuild/benchmark-populate-prefix/src"
  "/content/GPU-TASKS/build/_deps/benchmark-subbuild/benchmark-populate-prefix/src/benchmark-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/content/GPU-TASKS/build/_deps/benchmark-subbuild/benchmark-populate-prefix/src/benchmark-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/content/GPU-TASKS/build/_deps/benchmark-subbuild/benchmark-populate-prefix/src/benchmark-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
