package com.ravi.DeepNNND4J.AF;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.Sigmoid;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Created by 611445924 on 18/04/2017.
 */
public class SigmoidAF implements ActivationFunction {
    @Override
    public INDArray activate(INDArray input) {
        return Nd4j.getExecutioner().execAndReturn(new Sigmoid(input));
    }

    @Override
    public INDArray derivative(INDArray input) {
        return Nd4j.getExecutioner().execAndReturn(new Sigmoid(input).derivative());
    }
}
