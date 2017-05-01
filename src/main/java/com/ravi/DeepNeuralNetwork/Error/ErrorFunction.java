package com.ravi.DeepNeuralNetwork.Error;

/**
 * Created by 611445924 on 08/03/2017.
 */
public interface ErrorFunction {
    public double error(double[] desiredOutputs, double[] actOutputs);
}
