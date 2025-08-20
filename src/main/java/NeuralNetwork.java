package main.java;

import java.util.Locale;
import java.util.*;

public class NeuralNetwork {

    private Matrix weightsInputHidden;
    private Matrix weightsHiddenOutput;
    private Matrix biasHidden;
    private Matrix biasOutput;

    private double learningRate;
    private double momentum; // النسبة الخاصة بالMomentum

    // Velocities for Momentum
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
        this.momentum = momentum;

        weightsInputHidden = new Matrix(inputSize, hiddenSize);
        weightsInputHidden.randomize();

        weightsHiddenOutput = new Matrix(hiddenSize, outputSize);
        weightsHiddenOutput.randomize();

        biasHidden = new Matrix(1, hiddenSize);
        biasHidden.randomize();

        biasOutput = new Matrix(1, outputSize);
        biasOutput.randomize();

        // Initialize velocities with zeros
        vWeightsInputHidden = new Matrix(inputSize, hiddenSize);
        vWeightsHiddenOutput = new Matrix(hiddenSize, outputSize);
        vBiasHidden = new Matrix(1, hiddenSize);
        vBiasOutput = new Matrix(1, outputSize);
    }

    public Matrix forwardPass(Matrix input) {
        hiddenLayerInput = input.multiply(weightsInputHidden).add(biasHidden);
        hiddenLayerOutput = ActivationFunctions.applySigmoid(hiddenLayerInput);

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

    // Backpropagation with Momentum
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

        Matrix deltaWeightsHiddenOutput = hiddenLayerOutput.transpose().multiply(outputError).multiply(learningRate);
        Matrix deltaWeightsInputHidden = input.transpose().multiply(hiddenGradient).multiply(learningRate);

        Matrix deltaBiasOutput = outputError.multiply(learningRate);
        Matrix deltaBiasHidden = hiddenGradient.multiply(learningRate);

        // Update velocities
        vWeightsHiddenOutput = vWeightsHiddenOutput.multiply(momentum).add(deltaWeightsHiddenOutput);
        vWeightsInputHidden = vWeightsInputHidden.multiply(momentum).add(deltaWeightsInputHidden);
        vBiasOutput = vBiasOutput.multiply(momentum).add(deltaBiasOutput);
        vBiasHidden = vBiasHidden.multiply(momentum).add(deltaBiasHidden);

        // Update weights with velocities
        weightsHiddenOutput = weightsHiddenOutput.subtract(vWeightsHiddenOutput);
        weightsInputHidden = weightsInputHidden.subtract(vWeightsInputHidden);
        biasOutput = biasOutput.subtract(vBiasOutput);
        biasHidden = biasHidden.subtract(vBiasHidden);
    }

    public void train(Matrix[] inputs, Matrix[] targets, int epochs, int batchSize) {
        Random rand = new Random();
        System.out.println("Training started...");
        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalLoss = 0;
            double totalAccuracy = 0;

            List<Integer> indices = new ArrayList<>();
            for (int i = 0; i < inputs.length; i++) indices.add(i);
            Collections.shuffle(indices, rand);

            for (int b = 0; b < inputs.length; b++) {
                int idx = indices.get(b);
                Matrix prediction = forwardPass(inputs[idx]);
                totalLoss += calculateCrossEntropyLoss(prediction, targets[idx]);
                totalAccuracy += calculateAccuracy(prediction, targets[idx]);

                backpropagation(inputs[idx], targets[idx]);
            }

            double avgLoss = totalLoss / inputs.length;
            double avgAccuracy = totalAccuracy / inputs.length;

            System.out.println(String.format(Locale.ENGLISH,
                    "Epoch %d: Loss = %.6f, Accuracy = %.2f%%",
                    epoch + 1, avgLoss, avgAccuracy * 100));
        }
    }

    public void test(Matrix[] testInputs, Matrix[] testTargets) {
        double totalLoss = 0;
        double totalAccuracy = 0;

        for (int i = 0; i < testInputs.length; i++) {
            Matrix prediction = forwardPass(testInputs[i]);
            totalLoss += calculateCrossEntropyLoss(prediction, testTargets[i]);
            totalAccuracy += calculateAccuracy(prediction, testTargets[i]);
        }

        double avgLoss = totalLoss / testInputs.length;
        double avgAccuracy = totalAccuracy / testInputs.length;

        System.out.printf(Locale.ENGLISH,
                "Test results: Loss = %.6f, Accuracy = %.2f%%%n",
                avgLoss, avgAccuracy * 100);
    }
}
