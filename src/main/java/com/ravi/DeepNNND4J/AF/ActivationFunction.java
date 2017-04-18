package com.ravi.DeepNNND4J.AF;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Created by 611445924 on 18/04/2017.
 */
public interface ActivationFunction {
    public INDArray activate(INDArray input);
    public INDArray derivative(INDArray input);
}
