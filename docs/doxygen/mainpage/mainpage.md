\mainpage Main Page

CNeural is a simple implementation of a deep neural network in C.

[CMake](https://cmake.org/) is the build system for this project, however a compiler (such as GCC or Clang) is sufficient.  
Documentation is generated using [doxygen](https://www.doxygen.nl/).

## Navigation
- The data structure layout can be found under `Data Structures`  
- Source code and its documentation under `Files`
- Each function has a description of itself, **parameters** and **return values**

## Requirements
- Tested on
    - MinGW 11.0 w64 with C11 (Windows 11 -amd64), with CMake 3.30.5
    - gcc 13.3.1 with C18 (Gentoo Linux -x86_64), with and without CMake
    - gcc 13.2.0 with C18 (Ubuntu -aarch64), with and without CMake
- CMake => 3.30

## Usage/Examples
- Provided is an example `main.c` under `/src`
```
$ git clone https://github.com/DaiDuongLe/CNeural.git
```
- **Build** the CMake project

  ```bash
    $ cd CNeural
    $ cmake .
    $ cmake --build .
    $ ./CNeural
  ```

- **Compile** with GCC

  ```bash
    $ cd CNeural/src
    $ gcc main.c CNeural.c CNeural_backpropagation.c -lm -o main
    $ ./main
  ```