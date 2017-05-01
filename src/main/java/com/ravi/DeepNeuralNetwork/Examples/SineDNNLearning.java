package com.ravi.DeepNeuralNetwork.Examples;

import com.ravi.DeepNeuralNetwork.AF.LinearAF;
import com.ravi.DeepNeuralNetwork.AF.SigmoidAF;
import com.ravi.DeepNeuralNetwork.Learning.LearningAlgorithm;
import com.ravi.DeepNeuralNetwork.Learning.OnlineLearning;
import com.ravi.DeepNeuralNetwork.NeuralNetwork;
import com.ravi.DeepNeuralNetwork.Training.BackPropagation;
import com.ravi.Utils.ArrayUtils;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by rc16956 on 12/04/2017.
 */
public class SineDNNLearning {
    public static void main(String[] args){
        double interval= 2*Math.PI/9;
        List<Double> values = new ArrayList<Double>();
        for(double i=0; i<2.5; i=i+0.01){
            values.add(i);
        }

        double[][] inputs = new double[values.size()][1];
        int i=0;
        for(Double d : values) {
            inputs[i][0] = d;

            //System.out.println("Input "+inputs[i][0]);
            //System.out.println("Output "+outputs[i][0]);
            i++;
        }

        ArrayUtils.shuffleArray(inputs);
        int testSize = (int) (inputs.length * 0.1);

        double[][] testData = new double[testSize][1];
        double[][] testOutputs = new double[testSize][1];
        double[][] trainingData = new double[inputs.length-testSize][1];
        double[][] trainingOutputs = new double[inputs.length-testSize][1];

        for(int t=0; t<testSize; t++){
            testData[t][0] = inputs[t][0];
            testOutputs[t][0] = Math.sin(2*Math.PI*inputs[t][0]);
        }

        for(int t=testSize; t<inputs.length; t++){
            trainingData[t-testSize][0] = inputs[t][0];
            trainingOutputs[t-testSize][0] = Math.sin(2*Math.PI*inputs[t][0]);
        }

        NeuralNetwork neuralNetwork = new NeuralNetwork();
        neuralNetwork.addLayer(new SigmoidAF(1.0), 1, 10);
        neuralNetwork.addLayer(new LinearAF(), 10, 1);

        LearningAlgorithm learningAlgorithm = new OnlineLearning(new BackPropagation(0.1, 0.01), neuralNetwork, 0.2);
        learningAlgorithm.validation(0.2, 10000);
        learningAlgorithm.stoppingCriteria(-1, 0.999);
        neuralNetwork = learningAlgorithm.train(trainingData, trainingOutputs);

        for(int t=0; t<testData.length; t++) {
            double[] output = neuralNetwork.getOutput(testData[t]);
            System.out.println("NN Output :"+output[0]);
            System.out.println("Act Output :"+testOutputs[t][0]);
        }
    }
}
