cmake_minimum_required(VERSION 3.7.0)

project(projet LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package( OpenCV REQUIRED )
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

file(
GLOB_RECURSE
source_files
"src/*.cpp"
)

file(
GLOB_RECURSE
header_files
"src/*.h"
"src/*.hpp"
)


message("Source files")
foreach(line IN LISTS source_files)
    message("${line}")
endforeach() 

message("header files")
foreach(line IN LISTS header_files)
    message("${line}")
endforeach() 

if (WIN32)
  #set(LINK_FLAGS_PERSO )
else (WIN32)
  #set(LINK_FLAGS_PERSO -lm -ldl)
  add_definitions( -Wall -Wextra -Wno-comment -Wno-unused-parameter -Wno-unused-function -Wno-unused-variable -DCORRECTION -g -DCMAKE_C_FLAGS=-fopenmp -DCMAKE_EXE_LINKER_FLAGS=-fopenmp)
endif (WIN32)

add_executable( 
  projet
  ${source_files}
  ${header_files}
)

target_link_libraries(projet PUBLIC OpenMP::OpenMP_CXX ${OpenCV_LIBS})
