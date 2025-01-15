cmake_minimum_required(VERSION 3.30)
project(CNeural C)

set(CMAKE_C_STANDARD 11)

add_executable(CNeural src/main.c
        src/CNeural.h
        src/CNeural.c
#        CNeural_draw.c
        src/CNeural_backpropagation.c)
