# CNeural

CNeural is a simple implementation of a deep neural network in C.
 
[CMake](https://cmake.org/) is the build system for this project, however a compiler (such as GCC or Clang) is sufficient.  
Documentation is generated using [doxygen](https://www.doxygen.nl/). 



## Documentation

The documentation is available in `docs/doxygen/html` in HTML format under `index.html` [here](https://github.com/DaiDuongLe/CNeural/tree/main/) or RTF (Rich Text Format) in `docs/doxygen/rtf` under `refman.rtf` [here](https://github.com/DaiDuongLe/CNeural/tree/main/docs/doxygen/html). 

For RTF it is recommended to open using MS Word.
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
$ gcc CNeural.c CNeural_backpropagation.c -lm
```
## PVA Requirements
Name: Dai Duong Le  
Class: C1  
Grading Period: 2nd quarter  
Requirements: Implement a neural network in C  

### Steps:

- Understanding the structure of a neural network:
    - 3blue1brown Neural Networks Chapter 1-3:
        - []()
    - MIT 6.S191: Introduction to Deep Learning
        - []()
- Understanding the algorithm behind neural networks:
- Creating the data structure:
    - The data structure goes as follows: NeuralNetwork -> Layer -> Node (details under documentation)
- Implementation:
 

### Achievements:
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
    - Implement predition feature with proper label handling
    - Saving parameters to a standardized format
- Low priority:
    - GUI interface using Clay
