package com.ravi.Utils;

import java.util.Random;

/**
 * Created by rc16956 on 10/03/2017.
 */
public class ArrayUtils {
    public static void shuffleArray(double[][] ar)
    {
        // If running on Java 6 or older, use `new Random()` on RHS here
        Random rnd = new Random();
        for (int i = ar.length - 1; i > 0; i--)
        {
            int index = rnd.nextInt(i + 1);
            // Simple swap
            double[] a = ar[index];
            ar[index] = ar[i];
            ar[i] = a;
        }
    }
}
