package com.ravi.MultiLayerPerceptron.Training;

import com.ravi.MultiLayerPerceptron.MLP.MLPerceptron;

/**
 * Created by 611445924 on 08/03/2017.
 */
public interface TrainingAlgorithm {
    public void runEpoch(double[] inputs, double ekt, MLPerceptron perceptron, int t);
}
