package com.ravi.DeepNeuralNetwork;

import com.ravi.Utils.MatrixMath;

import java.util.List;

/**
 * Created by rc16956 on 10/04/2017.
 */
public class BackPropagation {
    List<NeuronLayer> network;


    public BackPropagation(List<NeuronLayer> network) {
        this.network = network;
    }

    public void train(double[] inputs, double[] outputs){
        double[] layerOutput = outputs;
        for(int i=network.size()-1; i>=0; i--){
            NeuronLayer layer = network.get(i);

            double[] actOutput = layer.getOutput(inputs);
            double[] error = MatrixMath.substract(layerOutput, actOutput);

            double[] deltaL = layer.getDelta(inputs, error);

            double[] previousLayerOutput = new double[inputs.length];
            if(i > 0) {
                previousLayerOutput = network.get(i - 1).getOutput(inputs);
            }
        }
    }
}
