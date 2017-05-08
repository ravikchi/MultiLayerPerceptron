package com.ravi.DeepNNND4J.Error;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Created by 611445924 on 08/05/2017.
 */
public class DefaultError implements ErrorFunction {
    public INDArray getError(INDArray desOutput, INDArray output){
        return desOutput.sub(output);
    }

    @Override
    public double getRMSError(INDArray desOutput, INDArray output) {
        INDArray error = getError(desOutput, output);
        double totalError = 0;
        for(int i=0; i<error.linearView().length(); i++){
            double localError = error.linearView().getDouble(i);
            totalError = totalError + localError*localError;
        }

        return Math.sqrt(totalError/error.linearView().length());
    }
}
