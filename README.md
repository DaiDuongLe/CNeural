# CNeural
CNeural is a simple implementation of a deep neural network in C.
 
[CMake](https://cmake.org/) is the build system for this project, however a compiler (such as GCC or Clang) is sufficient.  
Documentation is generated using [doxygen](https://www.doxygen.nl/).

## Documentation
The documentation is available in `docs/doxygen/html` in HTML format under `index.html` [here](https://github.com/DaiDuongLe/CNeural/tree/main/docs/doxygen/html) or RTF (Rich Text Format) in `docs/doxygen/rtf` under `refman.rtf` [here](https://github.com/DaiDuongLe/CNeural/tree/main/docs/doxygen/rtf). 

For RTF it is recommended to be opened using MS Word.

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
Grading Period: 4th quarter  
Requirements: Implement a neural network in C  

### Steps:

- Simplify Softmax + Cross Entropy derivatives
- Fix loss reporting (reset after each epoch)
- Train and validate results manually by changing step size and loss threshold until convergence


### Achievements:
- **Iris dataset convergence and successful prediction**
- Backpropagation for Softmax + Cross Entropy simplified
- Corrected loss reporting

### Next steps/goals (nice to haves, doesn't affect functionality):
- Parameter Initialization:
    - Add more initialization options to weights & biases
- Backpropagation and Optimization:
    - Convergence for MNIST
    - Stochastic Gradient Descent + Batching (improvement over raw GD)
    - Other optimizers (Adam, RMSprop, etc.)
    - Multithreading
    - Optimize Naive Matrix Multiplication
- Prediction:
    - Improve label handling
    - Saving network parameters to a standardized format
    - Argmax instead of Softmax for better readability
- Cleanup:
    - Reorganize code, versioning and redo technical documentation 
- Other:
    - GUI interface using Clay
    - Consolidate into a C library
