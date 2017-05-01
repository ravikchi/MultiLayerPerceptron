package com.ravi.DeepNeuralNetwork.Examples;

import com.ravi.DeepNeuralNetwork.AF.LinearAF;
import com.ravi.DeepNeuralNetwork.AF.SigmoidAF;
import com.ravi.DeepNeuralNetwork.Learning.LearningAlgorithm;
import com.ravi.DeepNeuralNetwork.NeuralNetwork;
import com.ravi.DeepNeuralNetwork.Training.BackPropagation;
import com.ravi.DeepNeuralNetwork.Learning.OnlineLearning;
import com.ravi.DeepNeuralNetwork.NeuronLayer;

import java.text.DecimalFormat;

/**
 * Created by rc16956 on 10/04/2017.
 */
public class XORDNNLearning {
    public static void main(String[] args){
        NeuralNetwork network = new NeuralNetwork();
        network.addLayer(new NeuronLayer(new SigmoidAF(1), 2, 2));
        network.addLayer(new NeuronLayer(new LinearAF(), 2, 1));

        double[][] inputs = {{1,0}, {0,1}, {1,1}, {0,0}};
        double[][] outputs = {{1}, {1}, {0}, {0}};

        LearningAlgorithm learning = new OnlineLearning(new BackPropagation(0.1, 0.01), network);
        learning.stoppingCriteria(-1, 0.000001);
        learning.train(inputs, outputs);
        DecimalFormat df2 = new DecimalFormat(".#####");

        for(int t=0; t<inputs.length; t++){
            double[] actualOutput = network.getOutput(inputs[t]);
            System.out.println("Actual Output : "+df2.format(actualOutput[0]));
        }
    }
}
