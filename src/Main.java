import java.io.*;
import java.util.*;

public class Main {
    public static void main(String[] args) throws IOException {
        // تحميل البيانات
        List<double[]> trainImages = loadImages("data/mnist_train.csv");
        List<double[]> trainLabels = loadLabels("data/mnist_train.csv");
        List<double[]> testImages = loadImages("data/mnist_test.csv");
        List<double[]> testLabels = loadLabels("data/mnist_test.csv");

        // تحقق من الأحجام
        System.out.println("Train images: " + trainImages.size());
        System.out.println("Train labels: " + trainLabels.size());
        System.out.println("Test images: " + testImages.size());
        System.out.println("Test labels: " + testLabels.size());

        if (trainImages.isEmpty() || testImages.isEmpty()) {
            System.out.println("❌ Error: لم يتم تحميل البيانات بشكل صحيح.");
            return;
        }

        // إنشاء الشبكة العصبية
        NeuralNetwork nn = new NeuralNetwork(784, 64, 10, 0.1);

        // التدريب (عدد epochs صغير للتجربة)
        for (int epoch = 1; epoch <= 3; epoch++) {
            System.out.println("Epoch " + epoch);
            for (int i = 0; i < trainImages.size(); i++) {
                nn.train(trainImages.get(i), trainLabels.get(i));
            }
        }

        // اختبار الدقة
        int correct = 0;
        for (int i = 0; i < testImages.size(); i++) {
            int predicted = nn.predict(testImages.get(i));
            int actual = getLabelIndex(testLabels.get(i));
            if (predicted == actual) correct++;
        }
        double accuracy = 100.0 * correct / testImages.size();
        System.out.printf("✅ Accuracy: %.2f%%\n", accuracy);
    }

    // دالة تحميل الصور
    private static List<double[]> loadImages(String path) throws IOException {
        List<double[]> images = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            String line;
            boolean firstLine = true;
            while ((line = br.readLine()) != null) {
                if (firstLine && line.contains("label")) {
                    firstLine = false; // تجاهل الـ header
                    continue;
                }
                String[] parts = line.split(",");
                if (parts.length < 785) continue; // سطر ناقص → تجاهله
                double[] inputs = new double[784];
                for (int j = 0; j < 784; j++) {
                    inputs[j] = Double.parseDouble(parts[j + 1]) / 255.0;
                }
                images.add(inputs);
            }
        }
        return images;
    }

    // دالة تحميل الـ labels
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

    // تحويل one-hot → index
    private static int getLabelIndex(double[] label) {
        for (int i = 0; i < label.length; i++) {
            if (label[i] == 1.0) return i;
        }
        return -1;
    }
}
