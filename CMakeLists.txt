cmake_minimum_required(VERSION 3.7.0)

project(projet LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package( OpenCV REQUIRED )

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
  add_definitions( -Wall -Wextra -Wno-comment -Wno-unused-parameter -Wno-unused-function -Wno-unused-variable -DCORRECTION)
endif (WIN32)

add_executable( 
  projet
  ${source_files}
  ${header_files}
)

target_link_libraries(projet ${OpenCV_LIBS})
