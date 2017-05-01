package com.ravi.DeepNNND4J.Examples;

import com.ravi.DeepNNND4J.AF.LinearAF;
import com.ravi.DeepNNND4J.AF.SigmoidAF;
import com.ravi.DeepNNND4J.Learning.LearningAlgorithm;
import com.ravi.DeepNNND4J.Learning.OnlineLearning;
import com.ravi.DeepNNND4J.NeuralNetwork;
import com.ravi.DeepNNND4J.Training.BackPropagation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Created by 611445924 on 18/04/2017.
 */
public class LearningTest {
    public static void main(String[] args){
        INDArray inputs = Nd4j.create(new double[]{1,0,1,0,0,1,1,0}, new int[]{2,4}, 'c');
        INDArray desOutputs = Nd4j.create(new double[]{1, 1, 0, 0}, new int[]{1, 4});

        NeuralNetwork network = new NeuralNetwork();
        network.addLayer(new SigmoidAF(), 2,2);
        network.addLayer(new LinearAF(), 2, 1);

        LearningAlgorithm learningAlgorithm = new OnlineLearning(network, new BackPropagation(0.1, 0.01));
        network = learningAlgorithm.train(inputs, desOutputs);

        for(int i=0; i<inputs.columns(); i++) {
            INDArray input = inputs.getColumn(i);
            System.out.println(input);

            INDArray output = network.getOutput(input);
            System.out.println("Output :"+ output);
        }
    }
}
