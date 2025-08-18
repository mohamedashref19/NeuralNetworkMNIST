import java.util.Random;

public class NeuralNetwork {
    private int inputSize;
    private int hiddenSize;
    private int outputSize;
    private double[][] weightsInputHidden;
    private double[][] weightsHiddenOutput;
    private double[] biasHidden;
    private double[] biasOutput;
    private double learningRate = 0.01;
    private Random rand = new Random();

    public NeuralNetwork(int inputSize, int hiddenSize, int outputSize, double v) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;

        weightsInputHidden = new double[inputSize][hiddenSize];
        weightsHiddenOutput = new double[hiddenSize][outputSize];
        biasHidden = new double[hiddenSize];
        biasOutput = new double[outputSize];

        initWeights();
    }

    private void initWeights() {
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                weightsInputHidden[i][j] = rand.nextGaussian() * 0.01;
            }
        }
        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                weightsHiddenOutput[i][j] = rand.nextGaussian() * 0.01;
            }
            biasHidden[i] = 0.0;
        }
        for (int i = 0; i < outputSize; i++) {
            biasOutput[i] = 0.0;
        }
    }

    private double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    private double sigmoidDerivative(double x) {
        return x * (1 - x);
    }

    private double[] softmax(double[] x) {
        double max = Double.NEGATIVE_INFINITY;
        for (double v : x) if (v > max) max = v;

        double sum = 0.0;
        double[] exp = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            exp[i] = Math.exp(x[i] - max);
            sum += exp[i];
        }

        double[] out = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            out[i] = exp[i] / sum;
        }
        return out;
    }

    public double[] forward(double[] input) {
        double[] hidden = new double[hiddenSize];
        double[] output = new double[outputSize];

        // Input → Hidden
        for (int j = 0; j < hiddenSize; j++) {
            double sum = biasHidden[j];
            for (int i = 0; i < inputSize; i++) {
                sum += input[i] * weightsInputHidden[i][j];
            }
            hidden[j] = sigmoid(sum);
        }

        // Hidden → Output
        for (int k = 0; k < outputSize; k++) {
            double sum = biasOutput[k];
            for (int j = 0; j < hiddenSize; j++) {
                sum += hidden[j] * weightsHiddenOutput[j][k];
            }
            output[k] = sum;
        }

        return softmax(output);
    }

    public void train(double[] input, double[] target) {
        double[] hidden = new double[hiddenSize];
        double[] output = new double[outputSize];

        // Forward pass
        for (int j = 0; j < hiddenSize; j++) {
            double sum = biasHidden[j];
            for (int i = 0; i < inputSize; i++) {
                sum += input[i] * weightsInputHidden[i][j];
            }
            hidden[j] = sigmoid(sum);
        }

        for (int k = 0; k < outputSize; k++) {
            double sum = biasOutput[k];
            for (int j = 0; j < hiddenSize; j++) {
                sum += hidden[j] * weightsHiddenOutput[j][k];
            }
            output[k] = sum;
        }
        output = softmax(output);

        // Output error
        double[] outputError = new double[outputSize];
        for (int k = 0; k < outputSize; k++) {
            outputError[k] = target[k] - output[k];
        }

        // Hidden → Output weight updates
        for (int j = 0; j < hiddenSize; j++) {
            for (int k = 0; k < outputSize; k++) {
                weightsHiddenOutput[j][k] += learningRate * outputError[k] * hidden[j];
            }
        }
        for (int k = 0; k < outputSize; k++) {
            biasOutput[k] += learningRate * outputError[k];
        }

        // Hidden error
        double[] hiddenError = new double[hiddenSize];
        for (int j = 0; j < hiddenSize; j++) {
            double error = 0.0;
            for (int k = 0; k < outputSize; k++) {
                error += outputError[k] * weightsHiddenOutput[j][k];
            }
            hiddenError[j] = error * sigmoidDerivative(hidden[j]);
        }

        // Input → Hidden weight updates
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                weightsInputHidden[i][j] += learningRate * hiddenError[j] * input[i];
            }
        }
        for (int j = 0; j < hiddenSize; j++) {
            biasHidden[j] += learningRate * hiddenError[j];
        }
    }

    public int predict(double[] input) {
        double[] output = forward(input);
        int label = 0;
        double max = output[0];
        for (int i = 1; i < output.length; i++) {
            if (output[i] > max) {
                max = output[i];
                label = i;
            }
        }
        return label;
    }
}
