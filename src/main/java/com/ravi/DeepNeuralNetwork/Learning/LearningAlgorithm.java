package com.ravi.DeepNeuralNetwork.Learning;

import com.ravi.DeepNeuralNetwork.NeuralNetwork;

/**
 * Created by rc16956 on 12/04/2017.
 */
public interface LearningAlgorithm {
    public NeuralNetwork train(double[][] inputs, double[][] outputs);
    public void validation(double percentage, int earlyStopCount);
    public void stoppingCriteria(int numberOfEpoches, double minimumError);
}
