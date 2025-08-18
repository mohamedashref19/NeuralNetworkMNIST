import java.io.*;
import java.util.*;

public class DataLoader {

    // تحميل البيانات من ملف CSV
    public static List<double[]> loadData(String filePath) throws IOException {
        return loadCSV(filePath);
    }

    // قراءة ملف CSV مع تخطي الـ header
    public static List<double[]> loadCSV(String filePath) throws IOException {
        List<double[]> data = new ArrayList<>();
        BufferedReader br = new BufferedReader(new FileReader(filePath));
        String line;
        boolean firstLine = true;

        while ((line = br.readLine()) != null) {
            if (firstLine) {
                firstLine = false;
                continue; // skip header
            }
            String[] values = line.split(",");
            double[] row = new double[values.length];
            for (int i = 0; i < values.length; i++) {
                row[i] = Double.parseDouble(values[i]);

            }
            data.add(row);
        }

        br.close();
        return data;
    }

    // فصل المدخلات عن المخرجات (one-hot labels)
    public static void splitFeaturesAndLabels(List<double[]> dataset,
                                              double[][] features, double[][] labels) {
        for (int i = 0; i < dataset.size(); i++) {
            double[] row = dataset.get(i);
            int label = (int) row[0];

            // Normalize pixels
            for (int j = 1; j < row.length; j++) {
                features[i][j - 1] = row[j] / 255.0;
            }

            // One-hot encoding للـ label
            labels[i][label] = 1.0;
        }
    }
}
