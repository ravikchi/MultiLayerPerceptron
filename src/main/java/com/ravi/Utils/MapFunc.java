package com.ravi.Utils;

import com.ravi.AF.ActivationFunction;

import java.util.List;

/**
 * Created by 611445924 on 10/04/2017.
 */
public class MapFunc {
    public static double[] map(ActivationFunction af, double[] output, boolean activate){
        double[] returnOut = new double[output.length];
        for(int i=0; i<output.length; i++){
            double val = 0.0;
            if(activate){
                val = af.activate(output[i]);
            }else{
                val = af.derivative(output[i]);
            }
            returnOut[i] = val;
        }

        return returnOut;
    }
}
