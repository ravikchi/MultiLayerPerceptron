package com.ravi.Examples;

import com.ravi.AF.SigmoidAF;
import com.ravi.DeepNeuralNetwork.BackPropagation;
import com.ravi.DeepNeuralNetwork.NeuronLayer;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by rc16956 on 10/04/2017.
 */
public class XORDNNLearning {
    public static void main(String[] args){
        List<NeuronLayer> network = new ArrayList<NeuronLayer>();
        network.add(new NeuronLayer(new SigmoidAF(1), 2, 2));
        network.add(new NeuronLayer(new SigmoidAF(1), 2, 1));

        double[][] inputs = {{1,0}, {0,1}, {1,1}, {0,0}};
        double[][] outputs = {{1}, {1}, {0}, {0}};

        BackPropagation trainingAlgo = new BackPropagation(network, 0.01, 0.01);
        for(int t=0; t<inputs.length; t++) {
            trainingAlgo.train(inputs[t], outputs[t]);
        }
    }
}
