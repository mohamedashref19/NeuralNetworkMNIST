package main.java;

public class ActivationFunctions {

    public static double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    public static double sigmoidDerivative(double x) {
        double s = sigmoid(x);
        return s * (1 - s);
    }

    public static double relu(double x) {
        return Math.max(0, x);
    }

    public static double reluDerivative(double x) {
        return x > 0 ? 1 : 0;
    }

    public static Matrix applySigmoid(Matrix input) {
        Matrix result = new Matrix(input.getRows(), input.getCols());
        for (int i = 0; i < input.getRows(); i++) {
            for (int j = 0; j < input.getCols(); j++) {
                result.set(i, j, sigmoid(input.get(i, j)));
            }
        }
        return result;
    }

    public static Matrix applySigmoidDerivative(Matrix input) {
        Matrix result = new Matrix(input.getRows(), input.getCols());
        for (int i = 0; i < input.getRows(); i++) {
            for (int j = 0; j < input.getCols(); j++) {
                result.set(i, j, sigmoidDerivative(input.get(i, j)));
            }
        }
        return result;
    }

    public static Matrix applyReLU(Matrix input) {
        Matrix result = new Matrix(input.getRows(), input.getCols());
        for (int i = 0; i < input.getRows(); i++) {
            for (int j = 0; j < input.getCols(); j++) {
                result.set(i, j, relu(input.get(i, j)));
            }
        }
        return result;
    }

    public static Matrix applyReLUDerivative(Matrix input) {
        Matrix result = new Matrix(input.getRows(), input.getCols());
        for (int i = 0; i < input.getRows(); i++) {
            for (int j = 0; j < input.getCols(); j++) {
                result.set(i, j, reluDerivative(input.get(i, j)));
            }
        }
        return result;
    }

    // ✅ Softmax (stable)
    public static Matrix applySoftmax(Matrix input) {
        Matrix result = new Matrix(input.getRows(), input.getCols());

        for (int i = 0; i < input.getRows(); i++) {
            double maxVal = input.get(i, 0);
            for (int j = 1; j < input.getCols(); j++) {
                if (input.get(i, j) > maxVal) {
                    maxVal = input.get(i, j);
                }
            }

            double sum = 0;
            for (int j = 0; j < input.getCols(); j++) {
                sum += Math.exp(input.get(i, j) - maxVal);
            }

            for (int j = 0; j < input.getCols(); j++) {
                double value = Math.exp(input.get(i, j) - maxVal) / sum;
                result.set(i, j, value);
            }
        }
        return result;
    }
}
