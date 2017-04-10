package com.ravi.MultiLayerPerceptron.Examples;

import com.ravi.AF.LinearAF;
import com.ravi.AF.SigmoidAF;
import com.ravi.Error.RMSE;
import com.ravi.MultiLayerPerceptron.Learning.BatchLearning;
import com.ravi.MultiLayerPerceptron.Learning.LearningAlgorithm;
import com.ravi.MultiLayerPerceptron.MLP.MLPerceptron;
import com.ravi.MultiLayerPerceptron.Training.BackPropagration;
import com.ravi.Utils.ArrayUtils;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by rc16956 on 10/03/2017.
 */
public class SineLearningBatch {
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

        MLPerceptron perceptron = new MLPerceptron(new LinearAF(),new SigmoidAF(1.0), 20, 1, 1);

        LearningAlgorithm learn = new BatchLearning(0.2, new BackPropagration(0.1, 0.01), new RMSE(), perceptron);
        learn.train(trainingData, trainingOutputs);



        for(int t=0; t<testData.length; t++) {
            double[] output = perceptron.getOutput(testData[t]);
            System.out.println("NN Output :"+output[0]);
            System.out.println("Act Output :"+testOutputs[t][0]);
        }
    }
}
