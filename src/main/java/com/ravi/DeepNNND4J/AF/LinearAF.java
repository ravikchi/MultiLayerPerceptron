package com.ravi.DeepNNND4J.AF;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Created by 611445924 on 18/04/2017.
 */
public class LinearAF implements ActivationFunction {
    @Override
    public INDArray activate(INDArray input) {
        return input;
    }

    @Override
    public INDArray derivative(INDArray input) {
        return input.assign(1.0);
    }
}
