public class Utils {
    public static void printMatrix(Matrix m) {
        for (int i = 0; i < m.rows; i++) {
            for (int j = 0; j < m.cols; j++) {
                System.out.print(m.data[i][j] + " ");
            }
            System.out.println();
        }
        System.out.println("----");
    }
}
