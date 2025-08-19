package main.java;

import java.io.*;
import java.util.*;

public class Main {
    public static void main(String[] args) throws IOException {
        List<double[]> trainImages = loadImages("data/mnist_train.csv");
        List<double[]> trainLabels = loadLabels("data/mnist_train.csv");
        List<double[]> testImages = loadImages("data/mnist_test.csv");
        List<double[]> testLabels = loadLabels("data/mnist_test.csv");

        System.out.println("Train images: " + trainImages.size());
        System.out.println("Train labels: " + trainLabels.size());
        System.out.println("Test images: " + testImages.size());
        System.out.println("Test labels: " + testLabels.size());

        if (trainImages.isEmpty() || testImages.isEmpty()) {
            System.out.println("❌ Error: The data wasn't loaded correctly.");
            return;
        }

        Matrix[] trainInputs = new Matrix[trainImages.size()];
        Matrix[] trainTargets = new Matrix[trainLabels.size()];

        for (int i = 0; i < trainImages.size(); i++) {
            trainInputs[i] = new Matrix(1, 784);
            trainTargets[i] = new Matrix(1, 10);

            for (int j = 0; j < 784; j++) {
                trainInputs[i].set(0, j, trainImages.get(i)[j]);
            }

            for (int j = 0; j < 10; j++) {
                trainTargets[i].set(0, j, trainLabels.get(i)[j]);
            }
        }

        NeuralNetwork nn = new NeuralNetwork(784, 64, 10, 0.1);

        nn.train(trainInputs, trainTargets, 10, 32); // 10 epochs, batch size 32


        Matrix[] testInputs = new Matrix[testImages.size()];
        Matrix[] testTargets = new Matrix[testLabels.size()];

        for (int i = 0; i < testImages.size(); i++) {
            testInputs[i] = new Matrix(1, 784);
            testTargets[i] = new Matrix(1, 10);

            for (int j = 0; j < 784; j++) {
                testInputs[i].set(0, j, testImages.get(i)[j]);
            }

            for (int j = 0; j < 10; j++) {
                testTargets[i].set(0, j, testLabels.get(i)[j]);
            }
        }

        // الاختبار
        nn.test(testInputs, testTargets);
    }

    private static List<double[]> loadImages(String path) throws IOException {
        List<double[]> images = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            String line;
            boolean firstLine = true;
            while ((line = br.readLine()) != null) {
                if (firstLine && line.contains("label")) {
                    firstLine = false;
                    continue;
                }
                String[] parts = line.split(",");
                if (parts.length < 785) continue;
                double[] inputs = new double[784];
                for (int j = 0; j < 784; j++) {
                    inputs[j] = Double.parseDouble(parts[j + 1]) / 255.0;
                }
                images.add(inputs);
            }
        }
        return images;
    }

    private static List<double[]> loadLabels(String path) throws IOException {
        List<double[]> labels = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            String line;
            boolean firstLine = true;
            while ((line = br.readLine()) != null) {
                if (firstLine && line.contains("label")) {
                    firstLine = false;
                    continue;
                }
                String[] parts = line.split(",");
                if (parts.length < 785) continue;
                double[] label = new double[10];
                int digit = Integer.parseInt(parts[0]);
                label[digit] = 1.0;
                labels.add(label);
            }
        }
        return labels;
    }
}