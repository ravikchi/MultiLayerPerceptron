package com.ravi.Utils;

import com.ravi.AF.ActivationFunction;

import java.util.List;

/**
 * Created by 611445924 on 10/04/2017.
 */
public class MapFunc {
    public static double[] map(ActivationFunction af, double[] output){
        double[] returnOut = new double[output.length];
        for(int i=0; i<output.length; i++){
            returnOut[i] = af.activate(output[i]);
        }

        return returnOut;
    }
}
