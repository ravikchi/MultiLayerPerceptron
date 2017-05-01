package com.ravi.DeepNeuralNetwork.Examples;

import com.ravi.DeepNeuralNetwork.AF.StepFunction;
import com.ravi.DeepNeuralNetwork.NeuronLayer;

/**
 * Created by rc16956 on 10/04/2017.
 */
public class XORDNN {

    public static void main(String[] args){
        NeuronLayer layer1 = new NeuronLayer(new StepFunction(), 2, 2);
        double[][] weightLayer1 = new double[2][2];
        weightLayer1[0][0] = 1;
        weightLayer1[0][1] = 1;

        weightLayer1[1][0] = 1;
        weightLayer1[1][1] = 1;

        double[] biasLayer1 = new double[2];
        biasLayer1[0] = -1.5;
        biasLayer1[1] = -0.5;

        layer1.setWeights(weightLayer1);
        layer1.setBias(biasLayer1);

        NeuronLayer layer2 = new NeuronLayer(new StepFunction(), 2, 1);
        double[][] weightLayer2 = new double[1][2];
        weightLayer2[0][0] = -2;
        weightLayer2[0][1] = 1;

        double[] biasLayer2 = new double[1];
        biasLayer2[0] = -0.5;

        layer2.setWeights(weightLayer2);
        layer2.setBias(biasLayer2);

        double[][] inputs = {{1,0}, {0,1}, {1,1}, {0,0}};

        for(int t=0; t<inputs.length; t++) {
            double[] outLayer1 = layer1.getOutput(inputs[t]);
            double[] finalOutput = layer2.getOutput(outLayer1);

            for (int i = 0; i < finalOutput.length; i++) {
                System.out.println(finalOutput[i]);
            }
        }
    }
}
