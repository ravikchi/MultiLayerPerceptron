package com.ravi.DeepNNND4J.Examples;

import com.ravi.DeepNNND4J.AF.LinearAF;
import com.ravi.DeepNNND4J.AF.SigmoidAF;
import com.ravi.DeepNNND4J.Learning.LearningAlgorithm;
import com.ravi.DeepNNND4J.Learning.OnlineLearning;
import com.ravi.DeepNNND4J.NeuralNetwork;
import com.ravi.DeepNNND4J.Training.BackPropagation;
import com.ravi.Utils.MNIST;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Created by ravik on 06/05/2017.
 */
public class TestAutoEncoder {
    public static void main(String[] args){
        MNIST mnist = new MNIST(1000, "C:\\Users\\ravik\\Downloads\\train.csv", "C:\\Users\\ravik\\Downloads\\test.csv");

        NeuralNetwork network = new NeuralNetwork();
        network.addLayer(new SigmoidAF(), 784, 100);
        network.addLayer(new SigmoidAF(), 100, 784);

        LearningAlgorithm learningAlgorithm = new OnlineLearning(network, new BackPropagation(0.1, 0.01));
        learningAlgorithm.train(mnist.getTrainingInput(), mnist.getTrainingInput());

        for(int i=0; i<mnist.getTestInput().columns(); i++) {
            INDArray input = mnist.getTestInput().getColumn(i);
            System.out.println("Desired Output "+mnist.getTestOutput().getColumn(i));

            INDArray output = network.getOutput(input);
            System.out.println("Output :"+ output);
        }
    }
}
