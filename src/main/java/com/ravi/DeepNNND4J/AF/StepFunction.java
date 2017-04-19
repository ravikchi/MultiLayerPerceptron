package com.ravi.DeepNNND4J.AF;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Created by 611445924 on 18/04/2017.
 */
public class StepFunction implements ActivationFunction {
    @Override
    public INDArray activate(INDArray input) {
        for(int i=0; i<input.linearView().length(); i++){
            if(input.linearView().getDouble(i) >= 0){
                input.linearView().putScalar(i, 1.0);
            }else{
                input.linearView().putScalar(i, 0.0);
            }
        }
        return input;
    }

    @Override
    public INDArray derivative(INDArray input) {
        return null;
    }


}
