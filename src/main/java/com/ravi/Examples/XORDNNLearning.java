package com.ravi.Examples;

import com.ravi.AF.SigmoidAF;
import com.ravi.DeepNeuralNetwork.Learning.LearningAlgorithm;
import com.ravi.DeepNeuralNetwork.NeuralNetwork;
import com.ravi.DeepNeuralNetwork.Training.BackPropagation;
import com.ravi.DeepNeuralNetwork.Learning.BatchLearning;
import com.ravi.DeepNeuralNetwork.NeuronLayer;

/**
 * Created by rc16956 on 10/04/2017.
 */
public class XORDNNLearning {
    public static void main(String[] args){
        NeuralNetwork network = new NeuralNetwork();
        network.addLayer(new NeuronLayer(new SigmoidAF(1), 2, 2));
        network.addLayer(new NeuronLayer(new SigmoidAF(1), 2, 1));

        double[][] inputs = {{1,0}, {0,1}, {1,1}, {0,0}};
        double[][] outputs = {{1}, {1}, {0}, {0}};

        LearningAlgorithm learning = new BatchLearning(new BackPropagation(0.1, 0.01), network);
        for(int t=0; t<100; t++) {
            learning.epoch(inputs, outputs);
        }
    }
}
