# CNeural
CNeural is a simple implementation of a deep neural network in C.
 
[CMake](https://cmake.org/) is the build system for this project, however a compiler (such as GCC or Clang) is sufficient.  
Documentation is generated using [doxygen](https://www.doxygen.nl/).

## Documentation
The documentation is available in `docs/doxygen/html` in HTML format under `index.html` [here](https://github.com/DaiDuongLe/CNeural/tree/main/docs/doxygen/html) or RTF (Rich Text Format) in `docs/doxygen/rtf` under `refman.rtf` [here](https://github.com/DaiDuongLe/CNeural/tree/main/docs/doxygen/rtf). 

For RTF it is recommended to open using MS Word.

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

## PVA Requirements
Name: Dai Duong Le  
Class: C1  
Grading Period: 2nd quarter  
Requirements: Implement a neural network in C  

### Steps:

- Understanding the structure of a neural network:
    - 3blue1brown Neural Networks Chapter 1-3:
        - [3b1b Neural Networks](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
    - MIT 6.S191: Introduction to Deep Learning 
        - [Lecture 1](https://youtu.be/ErnWZxJovaM?feature=shared)
- Understanding the algorithm behind neural networks:
    - 3blue1brown Neural Networks Chapter 4:
        - [Backpropagation Calculus](https://youtu.be/tIeHLnjs5U8)
    - Sebastian Lague
        - [How to Create a Neural Network (and Train it to Identify Doodles)](https://youtu.be/hfMk-kjRv4c?si=cNRpGwG0rPGNpvgm)
    - Artem Kirsanov
        - [The Most Important Algorithm in Machine Learning](https://youtu.be/SmZmBKc7Lrs?si=M6IoTvXO8Qw2Ehmq) 
- Creating the data structure and implementation:
    - The data structure goes as follows: NeuralNetwork -> Layer -> Node (more details in documentation)

### Achievements:
- Understanding neural networks on a fundamental level
- Neural Network data structure
- Configurable Forward propagation 
- Increased proficiency in C and documentation using doxygen.  

### Next steps/goals:
- Parameter Initialization:
    - Improve random weight init
    - Add more initialization options to weights & biases
- Backpropagation:
    - Calculate gradient of hidden layers
    - Update weights
- Data import:
    - Import MNIST dataset
- Prediction:
    - Implement prediction feature with proper label handling
    - Saving network parameters to a standardized format
- Other:
    - GUI interface using Clay
    - Consolidate into a C library
