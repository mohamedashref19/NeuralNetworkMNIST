public class Utils {
    public static void printMatrix(Matrix m) {
        for (int i = 0; i < m.getRows(); i++) {
            for (int j = 0; j < m.getCols(); j++) {
                System.out.print(m.get(i, j) + " ");
            }
            System.out.println();
        }
        System.out.println("----");
    }
}
