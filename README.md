# NeuralNetworkMNIST

A Java-based neural network project for classifying handwritten digits from the MNIST dataset using CSV files.

## Project Structure
- NeuralNetworkMNIST/
- │
- ├── data/   ` #CSV files containing the MNIST dataset`
- ├── src/     `# Source code files`
- ├── .gitignore `# Git ignore file`
- ├── README.md `# Project documentation`
- └── NeuralNetworkMNIST.iml

## Features

- Implements a neural network in Java.
- Trains on MNIST handwritten digit data stored in CSV format.
- Supports both training and testing of the model.
- Modular and easy-to-extend structure.

## How to Run

1. Make sure you have Java installed (JDK 8+ recommended).
2. Clone the repository:
   ```bash
   git clone https://github.com/mohamedashref19/NeuralNetworkMNIST.git

## Navigate to the project folder and compile the source code:

1. cd NeuralNetworkMNIST/src
2. javac *.java

## Run the main class:
- java Main

## Dataset
The project uses the MNIST dataset stored in CSV files (data/ folder). Each row represents a flattened 28x28 image of a handwritten digit, followed by its label.

 