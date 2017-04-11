package com.ravi.DeepNeuralNetwork;

import com.ravi.Utils.MatrixMath;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by rc16956 on 10/04/2017.
 */
public class BackPropagation {
    private List<NeuronLayer> network;
    private double learningRate;
    private double alpha;


    public BackPropagation(List<NeuronLayer> network, double learningRate, double alpha) {
        this.network = network;
        this.learningRate = learningRate;
        this.alpha = alpha;
    }

    public void train(double[] inputs, double[] outputs){
        double[] layerOutput = outputs;
        Map<NeuronLayer, double[]> layerMap = new HashMap<NeuronLayer, double[]>();

        double[] curInput = inputs;
        for(int i=0; i<network.size(); i++){
            NeuronLayer layer = network.get(i);
            curInput = layer.getOutput(curInput);
            layerMap.put(layer, curInput);
        }

        double[] error = null;

        for(int i=network.size()-1; i>=0; i--){
            NeuronLayer layer = network.get(i);

            if(i>0) {
                curInput = layerMap.get(network.get(i - 1));
            }else{
                curInput = inputs;
            }
            double[] actOutput = layerMap.get(layer);

            if(i>=network.size()-1) {
                error = MatrixMath.substract(layerOutput, actOutput);
            }

            error = updateWeights(layer, curInput, error);
        }
    }

    private double[] updateWeights(NeuronLayer layer, double[] input, double[] error){
        double[] deltaL = layer.getDelta(input, error);

        double[][] oldDeltaWeights = layer.getOldWeights();
        double[][] deltaWeigths = layer.getWeights();
        for(int j=0; j<deltaL.length; j++){
            deltaWeigths[j] = MatrixMath.scalarProduct(input, deltaL[j] * learningRate);

            deltaWeigths[j] = MatrixMath.add(deltaWeigths[j], MatrixMath.scalarProduct(oldDeltaWeights[j], alpha));
        }

        layer.setOldWeights(MatrixMath.copy(deltaWeigths));
        layer.setWeights(MatrixMath.add(layer.getWeights(), deltaWeigths));

        double[] output = new double[deltaWeigths[0].length];
        for(int i=0; i<deltaWeigths.length; i++){
            for(int j=0; j<deltaWeigths[i].length; j++){
                output[j] = output[j] + deltaWeigths[i][j];
            }
        }
        return output;
    }
}
