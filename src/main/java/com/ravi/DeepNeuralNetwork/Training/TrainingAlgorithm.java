package com.ravi.DeepNeuralNetwork.Training;

import com.ravi.DeepNeuralNetwork.NeuronLayer;

/**
 * Created by rc16956 on 12/04/2017.
 */
public interface TrainingAlgorithm {
    public double[][] train(NeuronLayer layer, double[] input, double[] error);
}
