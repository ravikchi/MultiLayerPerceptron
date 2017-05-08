package com.ravi.DeepNNND4J.Error;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Created by 611445924 on 18/04/2017.
 */
public interface ErrorFunction {
    public INDArray getError(INDArray desOutput, INDArray output);
    public double getRMSError(INDArray desOutput, INDArray output);
}
