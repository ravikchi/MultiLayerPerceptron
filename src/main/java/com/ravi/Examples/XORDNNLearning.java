package com.ravi.Examples;

import com.ravi.AF.SigmoidAF;
import com.ravi.DeepNeuralNetwork.NeuronLayer;

/**
 * Created by rc16956 on 10/04/2017.
 */
public class XORDNNLearning {
    public static void main(String[] args){
        NeuronLayer layer1 = new NeuronLayer(new SigmoidAF(1), 2, 2);

        NeuronLayer layer2 = new NeuronLayer(new SigmoidAF(1), 2, 1);

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
