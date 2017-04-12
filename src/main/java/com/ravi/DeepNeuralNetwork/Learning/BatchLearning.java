package com.ravi.DeepNeuralNetwork.Learning;

import com.ravi.DeepNeuralNetwork.NeuralNetwork;
import com.ravi.DeepNeuralNetwork.Training.BackPropagation;
import com.ravi.DeepNeuralNetwork.Training.TrainingAlgorithm;

/**
 * Created by rc16956 on 12/04/2017.
 */
public class BatchLearning implements LearningAlgorithm {
    TrainingAlgorithm trainingAlgo;
    NeuralNetwork network;

    public BatchLearning(BackPropagation trainingAlgo, NeuralNetwork network) {
        this.trainingAlgo = trainingAlgo;
        this.network = network;
    }

    public void epoch(double[][] inputs, double[][] outputs){
        for(int j=0; j<inputs.length; j++){
            double[] input = inputs[j];
            
        }
    }
}
