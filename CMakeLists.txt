cmake_minimum_required(VERSION 3.28)
project(hdrnn DESCRIPTION "Hand-written digit recognition neural network" LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

add_executable(hdrnn main.c)
