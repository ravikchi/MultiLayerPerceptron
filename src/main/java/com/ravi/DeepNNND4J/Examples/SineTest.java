package com.ravi.DeepNNND4J.Examples;

import com.ravi.DeepNNND4J.AF.LinearAF;
import com.ravi.DeepNNND4J.AF.SigmoidAF;
import com.ravi.DeepNNND4J.Learning.EarlyStopCriteria;
import com.ravi.DeepNNND4J.Learning.LearningAlgorithm;
import com.ravi.DeepNNND4J.Learning.OnlineLearning;
import com.ravi.DeepNNND4J.NNetworkND4j;
import com.ravi.DeepNNND4J.Training.BackPropagation;
import com.ravi.Utils.ArrayUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by 611445924 on 18/04/2017.
 */
public class SineTest {
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

        INDArray trainingInput = Nd4j.create(trainingData);
        INDArray trainingOutput = Nd4j.create(trainingOutputs);

        NNetworkND4j network = new NNetworkND4j();
        network.addLayer(new SigmoidAF(), 1, 10);
        network.addLayer(new LinearAF(), 10, 1);

        LearningAlgorithm learningAlgorithm = new OnlineLearning(network, new BackPropagation(0.1, 0.01));
        learningAlgorithm.setEarlyStopCriteria(new EarlyStopCriteria(0.2, 5000, 0.000001, 1000));
        network = learningAlgorithm.train(trainingInput.transpose(), trainingOutput.transpose());

        INDArray testInput = Nd4j.create(testData);
        INDArray testOutput = Nd4j.create(testOutputs);

        for(int k=0; k<testInput.rows(); k++) {
            INDArray input = testInput.getRow(k);
            System.out.println(input);

            INDArray output = network.getOutput(input);
            System.out.println("NNOutput :"+ output);
            System.out.println("ActOutput :"+testOutput.getRow(k));
        }
    }
}
