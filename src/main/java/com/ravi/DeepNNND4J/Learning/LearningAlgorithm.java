package com.ravi.DeepNNND4J.Learning;

import com.ravi.DeepNNND4J.NNetworkND4j;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Created by 611445924 on 18/04/2017.
 */
public interface LearningAlgorithm {
    public NNetworkND4j train(INDArray inputs, INDArray outputs);
}
