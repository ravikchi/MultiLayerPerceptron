package com.ravi.Utils;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Created by 611445924 on 09/03/2017.
 */
public class MatrixMath {

    /**
     * a-b
     * @param a
     * @param b
     */
    public static double[][] substraction(double[][] a, double[][] b){
        INDArray aArray = Nd4j.create(a);
        INDArray bArray = Nd4j.create(b);

        return getArray(aArray.sub(bArray));
    }

    public static double[][] getArray(INDArray c){
        int rowSize = c.rows();
        int colSize = c.columns();
        double[][] output = new double[rowSize][colSize];
        for(int i=0; i<rowSize; i++){
            for(int j=0; j<colSize; j++){
                output[i][j]=c.getDouble(i, j);
            }
        }
        return output;
    }

    public static double scalarValue(double[] a){
        double c = 0.0;
        for(int i=0; i<a.length; i++){
            c = a[i]+ c;
        }
        return c;
    }

    public static double[] substract(double[] a, double[] b){
        double[] c = new double[a.length];
        for(int i=0; i<a.length; i++){
            c[i] = a[i]-b[i];
        }
        return c;
    }

    // return a random m-by-n matrix with values between 0 and 1
    public static double[][] random(int m, int n) {
        double[][] a = new double[m][n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                a[i][j] = Math.random();
        return a;
    }

    // return n-by-n identity matrix I
    public static double[][] identity(int n) {
        double[][] a = new double[n][n];
        for (int i = 0; i < n; i++)
            a[i][i] = 1;
        return a;
    }

    public static double[] get1DArray(INDArray a){
        double[] output = new double[a.rows()];
        for(int i=0; i<a.rows(); i++){
            output[i] = a.getDouble(i);
        }
        return output;
    }

    public static double[] scalarProduct(double[] a, double s){
        double[] c = new double[a.length];
        for(int i=0; i<a.length; i++){
            c[i] = a[i] * s;
        }

        return c;
    }

    public static double[] arrayProduct(double[] a, double[] b){
        double[] c = new double[a.length];
        for(int i=0; i<a.length; i++){
            c[i] = a[i] * b[i];
        }

        return c;
    }

    // return x^T y
    public static double dot(double[] x, double[] y) {
        if (x.length != y.length) throw new RuntimeException("Illegal vector dimensions.");
        double sum = 0.0;
        for (int i = 0; i < x.length; i++)
            sum += x[i] * y[i];
        return sum;
    }

    // return B = A^T
    public static double[][] transpose(double[][] a) {
        int m = a.length;
        int n = a[0].length;
        double[][] b = new double[n][m];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                b[j][i] = a[i][j];
        return b;
    }

    public static double[] add(double[] a, double[] b){
        int m = a.length;
        double[] c = new double[m];
        for(int i=0; i<m; i++){
            c[i] = a[i]+b[i];
        }

        return c;
    }

    // return c = a + b
    public static double[][] add(double[][] a, double[][] b) {
        int m = a.length;
        int n = a[0].length;
        double[][] c = new double[m][n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                c[i][j] = a[i][j] + b[i][j];
        return c;
    }

    // return c = a - b
    public static double[][] subtract(double[][] a, double[][] b) {
        int m = a.length;
        int n = a[0].length;
        double[][] c = new double[m][n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                c[i][j] = a[i][j] - b[i][j];
        return c;
    }

    // return c = a * b
    public static double[][] multiply(double[][] a, double[][] b) {
        int m1 = a.length;
        int n1 = a[0].length;
        int m2 = b.length;
        int n2 = b[0].length;
        if (n1 != m2) throw new RuntimeException("Illegal matrix dimensions.");
        double[][] c = new double[m1][n2];
        for (int i = 0; i < m1; i++)
            for (int j = 0; j < n2; j++)
                for (int k = 0; k < n1; k++)
                    c[i][j] += a[i][k] * b[k][j];
        return c;
    }

    // matrix-vector multiplication (y = A * x)
    public static double[] multiply(double[][] a, double[] x) {
        int m = a.length;
        int n = a[0].length;
        if (x.length != n) throw new RuntimeException("Illegal matrix dimensions.");
        double[] y = new double[m];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                y[i] += a[i][j] * x[j];
        return y;
    }


    // vector-matrix multiplication (y = x^T A)
    public static double[] multiply(double[] x, double[][] a) {
        int m = a.length;
        int n = a[0].length;
        if (x.length != m) throw new RuntimeException("Illegal matrix dimensions.");
        double[] y = new double[n];
        for (int j = 0; j < n; j++)
            for (int i = 0; i < m; i++)
                y[j] += a[i][j] * x[i];
        return y;
    }

    public static double[][] copy(double[][] a){
        if(inValidMatrix(a)){
            return null;
        }
        double[][] b = new double[a.length][a[0].length];
        for(int i=0; i<a.length; i++) {
            for (int j = 0; j < a[i].length; j++) {
                b[i][j] = a[i][j];
            }
        }

        return b;
    }

    public static double[] copy(double[] a){
        double[] c = new double[a.length];
        for(int i=0; i<a.length; i++){
            c[i] = a[i];
        }
        return c;
    }

    private static boolean inValidMatrix(double[][] a){
        if(a== null ){
            return true;
        }
        return false;
    }

    private static boolean inValidMatrices(double[][] a, double[][] b){
        if(a== null && b== null ){
            return true;
        }
        if(a.length != b.length){
            if(a[0].length != b[0].length){
                return true;
            }
        }

        return false;
    }
}
