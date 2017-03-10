package com.ravi.BackProp.Training;

import com.ravi.BackProp.MLP.MLPerceptron;

/**
 * Created by 611445924 on 08/03/2017.
 */
public interface TrainingAlgorithm {
    public void runEpoch(double[] inputs, double ekt, MLPerceptron perceptron, int t);
}
