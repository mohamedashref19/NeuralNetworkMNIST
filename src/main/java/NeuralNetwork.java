package main.java;

import java.util.Locale;
import java.util.*;

public class NeuralNetwork {

    private Matrix weightsInputHidden;
    private Matrix weightsHiddenOutput;
    private Matrix biasHidden;
    private Matrix biasOutput;

    private double learningRate;
    private double initialLearningRate; // للـ learning rate decay
    private double momentum;

    private Matrix vWeightsInputHidden;
    private Matrix vWeightsHiddenOutput;
    private Matrix vBiasHidden;
    private Matrix vBiasOutput;

    private Matrix hiddenLayerInput;
    private Matrix hiddenLayerOutput;
    private Matrix outputLayerInput;
    private Matrix finalOutput;

    public NeuralNetwork(int inputSize, int hiddenSize, int outputSize, double learningRate, double momentum) {
        this.learningRate = learningRate;
        this.initialLearningRate = learningRate;
        this.momentum = momentum;

        initializeWeightsXavier(inputSize, hiddenSize, outputSize);

        vWeightsInputHidden = new Matrix(inputSize, hiddenSize);
        vWeightsHiddenOutput = new Matrix(hiddenSize, outputSize);
        vBiasHidden = new Matrix(1, hiddenSize);
        vBiasOutput = new Matrix(1, outputSize);
    }

    private void initializeWeightsXavier(int inputSize, int hiddenSize, int outputSize) {
        // Xavier init للـ input-hidden weights
        double limitIH = Math.sqrt(6.0 / (inputSize + hiddenSize));
        weightsInputHidden = new Matrix(inputSize, hiddenSize);
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                weightsInputHidden.set(i, j, (Math.random() * 2 - 1) * limitIH);
            }
        }

        double limitHO = Math.sqrt(6.0 / (hiddenSize + outputSize));
        weightsHiddenOutput = new Matrix(hiddenSize, outputSize);
        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                weightsHiddenOutput.set(i, j, (Math.random() * 2 - 1) * limitHO);
            }
        }

        biasHidden = new Matrix(1, hiddenSize);
        for (int j = 0; j < hiddenSize; j++) {
            biasHidden.set(0, j, (Math.random() - 0.5) * 0.01);
        }

        biasOutput = new Matrix(1, outputSize);
        for (int j = 0; j < outputSize; j++) {
            biasOutput.set(0, j, (Math.random() - 0.5) * 0.01);
        }
    }

    public Matrix forwardPass(Matrix input) {
        // Hidden layer
        hiddenLayerInput = input.multiply(weightsInputHidden).add(biasHidden);
        hiddenLayerOutput = ActivationFunctions.applySigmoid(hiddenLayerInput);

        // Output layer
        outputLayerInput = hiddenLayerOutput.multiply(weightsHiddenOutput).add(biasOutput);
        finalOutput = ActivationFunctions.applySoftmax(outputLayerInput);

        return finalOutput;
    }

    public double calculateCrossEntropyLoss(Matrix predicted, Matrix actual) {
        double epsilon = 1e-12;
        double loss = 0.0;
        for (int j = 0; j < predicted.getCols(); j++) {
            double y = actual.get(0, j);
            double p = Math.max(epsilon, Math.min(1 - epsilon, predicted.get(0, j)));
            loss += -y * Math.log(p);
        }
        return loss;
    }

    public double calculateAccuracy(Matrix predicted, Matrix actual) {
        int predictedClass = 0;
        int actualClass = 0;

        for (int j = 1; j < predicted.getCols(); j++) {
            if (predicted.get(0, j) > predicted.get(0, predictedClass)) {
                predictedClass = j;
            }
        }

        for (int j = 1; j < actual.getCols(); j++) {
            if (actual.get(0, j) > actual.get(0, actualClass)) {
                actualClass = j;
            }
        }

        return (predictedClass == actualClass) ? 1.0 : 0.0;
    }

    public void backpropagation(Matrix input, Matrix target) {
        Matrix prediction = forwardPass(input);

        Matrix outputError = prediction.subtract(target);

        Matrix hiddenError = outputError.multiply(weightsHiddenOutput.transpose());

        Matrix hiddenGradient = ActivationFunctions.applySigmoidDerivative(hiddenLayerInput);
        for (int i = 0; i < hiddenGradient.getRows(); i++) {
            for (int j = 0; j < hiddenGradient.getCols(); j++) {
                hiddenGradient.set(i, j, hiddenGradient.get(i, j) * hiddenError.get(i, j));
            }
        }

        Matrix gradWeightsHiddenOutput = hiddenLayerOutput.transpose().multiply(outputError);
        Matrix gradWeightsInputHidden = input.transpose().multiply(hiddenGradient);
        Matrix gradBiasOutput = outputError;
        Matrix gradBiasHidden = hiddenGradient;

        vWeightsHiddenOutput = vWeightsHiddenOutput.multiply(momentum)
                .subtract(gradWeightsHiddenOutput.multiply(learningRate));

        vWeightsInputHidden = vWeightsInputHidden.multiply(momentum)
                .subtract(gradWeightsInputHidden.multiply(learningRate));

        vBiasOutput = vBiasOutput.multiply(momentum)
                .subtract(gradBiasOutput.multiply(learningRate));

        vBiasHidden = vBiasHidden.multiply(momentum)
                .subtract(gradBiasHidden.multiply(learningRate));

        // === Update Parameters: w = w + v ===
        weightsHiddenOutput = weightsHiddenOutput.add(vWeightsHiddenOutput);
        weightsInputHidden = weightsInputHidden.add(vWeightsInputHidden);
        biasOutput = biasOutput.add(vBiasOutput);
        biasHidden = biasHidden.add(vBiasHidden);
    }

    public void updateLearningRate(int epoch) {
        // Exponential decay كل 10 epochs
        if (epoch % 10 == 0 && epoch > 0) {
            learningRate = initialLearningRate * Math.pow(0.95, epoch / 10);
            System.out.println("Learning rate updated to: " + String.format(Locale.ENGLISH, "%.6f", learningRate));
        }
    }

    public void train(Matrix[] inputs, Matrix[] targets, int epochs, int batchSize) {
        Random rand = new Random();
        System.out.println("Training started...");
        System.out.println("Initial Learning Rate: " + learningRate);
        System.out.println("Momentum: " + momentum);
        System.out.println("Batch Size: " + batchSize);
        System.out.println("=====================================");

        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalLoss = 0;
            double totalAccuracy = 0;

            List<Integer> indices = new ArrayList<>();
            for (int i = 0; i < inputs.length; i++) indices.add(i);
            Collections.shuffle(indices, rand);

            for (int b = 0; b < inputs.length; b += batchSize) {
                int batchEnd = Math.min(b + batchSize, inputs.length);

                for (int i = b; i < batchEnd; i++) {
                    int idx = indices.get(i);
                    Matrix prediction = forwardPass(inputs[idx]);
                    totalLoss += calculateCrossEntropyLoss(prediction, targets[idx]);
                    totalAccuracy += calculateAccuracy(prediction, targets[idx]);

                    backpropagation(inputs[idx], targets[idx]);
                }
            }

            double avgLoss = totalLoss / inputs.length;
            double avgAccuracy = totalAccuracy / inputs.length;

            System.out.println(String.format(Locale.ENGLISH,
                    "Epoch %d: Loss = %.6f, Accuracy = %.2f%%, LR = %.6f",
                    epoch + 1, avgLoss, avgAccuracy * 100, learningRate));

            // Update learning rate
            updateLearningRate(epoch + 1);
        }
        System.out.println("=====================================");
        System.out.println("Training completed!");
    }

    public void test(Matrix[] testInputs, Matrix[] testTargets) {
        double totalLoss = 0;
        double totalAccuracy = 0;

        System.out.println("Testing started...");

        for (int i = 0; i < testInputs.length; i++) {
            Matrix prediction = forwardPass(testInputs[i]);
            totalLoss += calculateCrossEntropyLoss(prediction, testTargets[i]);
            totalAccuracy += calculateAccuracy(prediction, testTargets[i]);
        }

        double avgLoss = totalLoss / testInputs.length;
        double avgAccuracy = totalAccuracy / testInputs.length;

        System.out.println("=====================================");
        System.out.printf(Locale.ENGLISH,
                "Test Results: Loss = %.6f, Accuracy = %.2f%%%n",
                avgLoss, avgAccuracy * 100);
        System.out.println("=====================================");
    }

    public void printNetworkInfo() {
        System.out.println("Neural Network Architecture:");
        System.out.println("Input Layer: " + weightsInputHidden.getRows() + " neurons");
        System.out.println("Hidden Layer: " + weightsInputHidden.getCols() + " neurons (Sigmoid)");
        System.out.println("Output Layer: " + weightsHiddenOutput.getCols() + " neurons (Softmax)");
        System.out.println("Learning Rate: " + learningRate);
        System.out.println("Momentum: " + momentum);
        System.out.println("=====================================");
    }
}