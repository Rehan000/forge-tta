# ESP-IDF top-level project file. Requires ESP-IDF v5.x.
#   idf.py set-target esp32s3
#   idf.py build flash monitor
cmake_minimum_required(VERSION 3.16)
include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(tta_edge)
