package com.ravi.DeepNNND4J.Error;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Created by 611445924 on 18/04/2017.
 */
public class ErrorFunction {
    public double getError(INDArray desOutput, INDArray output){
        INDArray input = desOutput.sub(output);
        double totalError = 0.0;
        for(int i=0; i<input.linearView().length(); i++){
            double val = input.linearView().getDouble(i);
            totalError = totalError + val * val;
        }
        return totalError/input.linearView().length();
    }
}
