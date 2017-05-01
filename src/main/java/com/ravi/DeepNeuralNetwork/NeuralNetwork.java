package com.ravi.DeepNeuralNetwork;

import com.ravi.DeepNeuralNetwork.AF.ActivationFunction;
import com.ravi.Utils.MatrixMath;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by rc16956 on 12/04/2017.
 */
public class NeuralNetwork {
    private List<NeuronLayer> layers = new ArrayList<NeuronLayer>();
    private Map<Integer, InputError> layerMap = new HashMap<Integer, InputError>();

    public void addLayer(NeuronLayer layer){
        layers.add(layer);
    }

    public void addLayer(ActivationFunction activationFunction, int numberOfInputs, int numberOfNeurons){
        layers.add(new NeuronLayer(activationFunction, numberOfInputs, numberOfNeurons));
    }

    public NeuralNetwork clone(){
        NeuralNetwork newNetwork = new NeuralNetwork();
        for(int i=0; i<size(); i++) {
            newNetwork.addLayer(layers.get(i).clone());
        }
        return newNetwork;
    }

    public double[] getOutput(double[] input){
        double[] curInput = input;
        layerMap.clear();
        for(int i=0; i<layers.size(); i++){
            NeuronLayer layer = layers.get(i);
            layerMap.put(i, new InputError(curInput, null));
            curInput = layer.getOutput(curInput);
        }
        return curInput;
    }

    public double[] getOutput(double[] input, int id){
        double[] curInput = input;
        if (id >= 0) {
            for (int i = 0; i < layers.size(); i++) {
                NeuronLayer layer = layers.get(i);
                curInput = layer.getOutput(curInput);
                if (i == id) {
                    break;
                }
            }
        }
        return curInput;
    }

    public double[] getError(double[] input, double[] error, int id){
        double[] curError = error;
        if(id>=layers.size()){
            return curError;
        }
        for(int i=layers.size()-1; i>=0; i--){
            NeuronLayer layer = layers.get(i);
            double[] delta = layer.getDelta(getOutput(input, i-1), curError);
            double[][] weightsT = MatrixMath.transpose(layer.getWeights());
            curError = MatrixMath.multiply(weightsT, delta);
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

    public NeuronLayer getLayer(int i){
        return layers.get(i);
    }

    public int size(){
        return layers.size();
    }

    class InputError{
        double[] input;
        double[] error;

        public InputError(double[] input, double[] error) {
            this.input = input;
            this.error = error;
        }

        public double[] getInput() {
            return input;
        }

        public void setInput(double[] input) {
            this.input = input;
        }
    }
}
