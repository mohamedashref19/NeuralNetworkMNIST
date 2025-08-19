import java.util.Locale;

public class NeuralNetwork {

    private Matrix weightsInputHidden;
    private Matrix weightsHiddenOutput;
    private Matrix biasHidden;
    private Matrix biasOutput;

    private double learningRate;

    private Matrix hiddenLayerInput;
    private Matrix hiddenLayerOutput;
    private Matrix outputLayerInput;
    private Matrix finalOutput;

    public NeuralNetwork(int inputSize, int hiddenSize, int outputSize, double learningRate) {
        this.learningRate = learningRate;

        weightsInputHidden = new Matrix(inputSize, hiddenSize);
        weightsInputHidden.randomize();
        weightsInputHidden = weightsInputHidden.multiply(0.5);

        weightsHiddenOutput = new Matrix(hiddenSize, outputSize);
        weightsHiddenOutput.randomize();
        weightsHiddenOutput = weightsHiddenOutput.multiply(0.5);

        biasHidden = new Matrix(1, hiddenSize);
        biasHidden.randomize();
        biasHidden = biasHidden.multiply(0.1);

        biasOutput = new Matrix(1, outputSize);
        biasOutput.randomize();
        biasOutput = biasOutput.multiply(0.1);
    }

    public Matrix forwardPass(Matrix input) {
        if (input.getRows() != 1) {
            throw new IllegalArgumentException("Input should be a row matrix (1 x n)");
        }

        hiddenLayerInput = input.multiply(weightsInputHidden).add(biasHidden);

        hiddenLayerOutput = ActivationFunctions.applySigmoid(hiddenLayerInput);

        outputLayerInput = hiddenLayerOutput.multiply(weightsHiddenOutput).add(biasOutput);

        finalOutput = ActivationFunctions.applySigmoid(outputLayerInput);

        return finalOutput;
    }

    public Matrix predict(Matrix input) {
        return forwardPass(input);
    }

    public double calculateLoss(Matrix predicted, Matrix actual) {
        if (predicted.getRows() != actual.getRows() || predicted.getCols() != actual.getCols()) {
            throw new IllegalArgumentException("Predicted and actual matrices must have same dimensions");
        }

        double totalLoss = 0;
        int totalElements = predicted.getRows() * predicted.getCols();

        for (int i = 0; i < predicted.getRows(); i++) {
            for (int j = 0; j < predicted.getCols(); j++) {
                double diff = predicted.get(i, j) - actual.get(i, j);
                totalLoss += diff * diff;
            }
        }

        return totalLoss / totalElements;
    }

    public double calculateAccuracy(Matrix predicted, Matrix actual) {
        if (predicted.getRows() != actual.getRows() || predicted.getCols() != actual.getCols()) {
            throw new IllegalArgumentException("Predicted and actual matrices must have same dimensions");
        }

        int correct = 0;
        int total = predicted.getRows();

        for (int i = 0; i < predicted.getRows(); i++) {

            int predictedClass = 0;
            int actualClass = 0;

            for (int j = 1; j < predicted.getCols(); j++) {
                if (predicted.get(i, j) > predicted.get(i, predictedClass)) {
                    predictedClass = j;
                }
            }

            for (int j = 1; j < actual.getCols(); j++) {
                if (actual.get(i, j) > actual.get(i, actualClass)) {
                    actualClass = j;
                }
            }

            if (predictedClass == actualClass) {
                correct++;
            }
        }

        return (double) correct / total;
    }

    public Matrix getWeightsInputHidden() { return weightsInputHidden; }
    public Matrix getWeightsHiddenOutput() { return weightsHiddenOutput; }
    public Matrix getBiasHidden() { return biasHidden; }
    public Matrix getBiasOutput() { return biasOutput; }
    public double getLearningRate() { return learningRate; }

    public void backpropagation(Matrix input, Matrix target) {
        Matrix prediction = forwardPass(input);

        Matrix outputError = target.subtract(prediction);

        Matrix outputGradient = ActivationFunctions.applySigmoidDerivative(outputLayerInput);
        // Element-wise multiplication
        for (int i = 0; i < outputGradient.getRows(); i++) {
            for (int j = 0; j < outputGradient.getCols(); j++) {
                double value = outputError.get(i, j) * outputGradient.get(i, j) * learningRate;
                outputGradient.set(i, j, value);
            }
        }

        Matrix hiddenOutputWeightsDelta = hiddenLayerOutput.transpose().multiply(outputGradient);

        weightsHiddenOutput = weightsHiddenOutput.add(hiddenOutputWeightsDelta);
        biasOutput = biasOutput.add(outputGradient);

        Matrix hiddenError = outputGradient.multiply(weightsHiddenOutput.transpose());

        Matrix hiddenGradient = ActivationFunctions.applySigmoidDerivative(hiddenLayerInput);
        // Element-wise multiplication
        for (int i = 0; i < hiddenGradient.getRows(); i++) {
            for (int j = 0; j < hiddenGradient.getCols(); j++) {
                double value = hiddenError.get(i, j) * hiddenGradient.get(i, j) * learningRate;
                hiddenGradient.set(i, j, value);
            }
        }

        Matrix inputHiddenWeightsDelta = input.transpose().multiply(hiddenGradient);

        weightsInputHidden = weightsInputHidden.add(inputHiddenWeightsDelta);
        biasHidden = biasHidden.add(hiddenGradient);
    }

    public void train(Matrix[] inputs, Matrix[] targets, int epochs) {
        if (inputs.length != targets.length) {
            throw new IllegalArgumentException("Number of inputs must equal number of targets");
        }

        System.out.println("Training started...");
        System.out.println("Number of samples: " + inputs.length);
        System.out.println("Number of Epochs: " + epochs);
        System.out.println("Learning rate: " + learningRate);
        System.out.println("------------------------");

        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalLoss = 0;
            double totalAccuracy = 0;

            for (int i = 0; i < inputs.length; i++) {

                Matrix prediction = forwardPass(inputs[i]);

                totalLoss += calculateLoss(prediction, targets[i]);
                totalAccuracy += calculateAccuracy(prediction, targets[i]);

                backpropagation(inputs[i], targets[i]);
            }

            double avgLoss = totalLoss / inputs.length;
            double avgAccuracy = totalAccuracy / inputs.length;

                System.out.println(String.format(Locale.ENGLISH,
                        "Epoch %d: Loss = %.6f, Accuracy = %.2f%%",
                        epoch + 1, avgLoss, avgAccuracy * 100));

        }

        System.out.println("Training finished!");
    }

    public void test(Matrix[] testInputs, Matrix[] testTargets) {
        if (testInputs.length != testTargets.length) {
            throw new IllegalArgumentException("Number of test inputs must equal number of test targets");
        }

        double totalLoss = 0;
        double totalAccuracy = 0;

        System.out.println("\nTesting started...");
        System.out.println("Number of test samples: " + testInputs.length);
        System.out.println("------------------------");

        for (int i = 0; i < testInputs.length; i++) {
            Matrix prediction = predict(testInputs[i]);
            totalLoss += calculateLoss(prediction, testTargets[i]);
            totalAccuracy += calculateAccuracy(prediction, testTargets[i]);
        }

        double avgLoss = totalLoss / testInputs.length;
        double avgAccuracy = totalAccuracy / testInputs.length;

        System.out.println(String.format(Locale.ENGLISH,
                "Test results: Loss = %.6f, Accuracy = %.2f%%",
                avgLoss, avgAccuracy * 100));
    }
    public void train(double[] input, double[] target) {
        Matrix inputMatrix = new Matrix(1, input.length);
        Matrix targetMatrix = new Matrix(1, target.length);

        for (int i = 0; i < input.length; i++) {
            inputMatrix.set(0, i, input[i]);
        }

        for (int i = 0; i < target.length; i++) {
            targetMatrix.set(0, i, target[i]);
        }

        backpropagation(inputMatrix, targetMatrix);
    }

    public int predict(double[] input) {
        Matrix inputMatrix = new Matrix(1, input.length);
        for (int i = 0; i < input.length; i++) {
            inputMatrix.set(0, i, input[i]);
        }

        Matrix output = forwardPass(inputMatrix);

        int maxIndex = 0;
        for (int i = 1; i < output.getCols(); i++) {
            if (output.get(0, i) > output.get(0, maxIndex)) {
                maxIndex = i;
            }
        }
        return maxIndex;
    }
}
