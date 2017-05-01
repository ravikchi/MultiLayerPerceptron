package com.ravi.DeepNNND4J.Learning;

import com.ravi.DeepNNND4J.NeuralNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Created by 611445924 on 18/04/2017.
 */
public interface LearningAlgorithm {
    public NeuralNetwork train(INDArray inputs, INDArray outputs);
    public void setEarlyStopCriteria(EarlyStopCriteria earlyStopCriteria);
}
