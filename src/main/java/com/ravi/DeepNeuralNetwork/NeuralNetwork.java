package com.ravi.DeepNeuralNetwork;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by rc16956 on 12/04/2017.
 */
public class NeuralNetwork {
    List<NeuronLayer> layers = new ArrayList<NeuronLayer>();

    public void addLayer(NeuronLayer layer){
        layers.add(layer);
    }

    public double[] getOutput(double[] input){
        double[] curInput = input;
        for(int i=0; i<layers.size(); i++){
            NeuronLayer layer = layers.get(i);
            curInput = layer.getOutput(curInput);
        }
        return curInput;
    }

    public double[] getOutput(double[] input, int id){
        double[] curInput = input;
        for(int i=0; i<layers.size(); i++){
            NeuronLayer layer = layers.get(i);
            curInput = layer.getOutput(curInput);
            if(i==id){
                break;
            }
        }
        return curInput;
    }

    public double[] revOutput(double[] input, double[] error, int id){
        double[] curError = error;
        for(int i=layers.size()-1; i>=0; i--){
            NeuronLayer layer = layers.get(i);
            curError = layer.getDelta(getOutput(input, i-1), curError);
            if(i==id){
                break;
            }
        }
        return curError;
    }

    public List<NeuronLayer> getLayers() {
        return layers;
    }

    public void setLayers(List<NeuronLayer> layers) {
        this.layers = layers;
    }
}
