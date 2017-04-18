package com.ravi.DeepNNND4J;

import com.ravi.DeepNNND4J.AF.ActivationFunction;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by 611445924 on 18/04/2017.
 */
public class NNetworkND4j {
    List<NNLayer> network = new ArrayList<NNLayer>();

    public void addLayer(ActivationFunction activationFunction, int numberOfInputs, int numberOfNeurons){
        network.add(new NNLayer(activationFunction, numberOfInputs, numberOfNeurons));
    }

    public void addLayer(NNLayer layer){
        network.add(layer);
    }

    public NNetworkND4j getClone(){
        NNetworkND4j newNetwork = new NNetworkND4j();
        for(NNLayer layer : network){
            newNetwork.addLayer(layer.clone());
        }

        return newNetwork;
    }

    public INDArray getOutput(INDArray input){
        INDArray curInput = input;
        for(int i=0; i<network.size(); i++){
            NNLayer layer = network.get(i);
            curInput = layer.getOutput(curInput);
        }

        return curInput;
    }

    public INDArray getOutput(INDArray input, int id){
        INDArray curInput = input;
        if(id<0){
            return curInput;
        }
        for(int i=0; i<network.size(); i++){
            NNLayer layer = network.get(i);
            curInput = layer.getOutput(curInput);
            if(id == i){
                break;
            }
        }

        return curInput;
    }

    public INDArray getError(INDArray input, INDArray output, int id){
        INDArray curError = output;
        if(id>=network.size()){
            return curError;
        }
        for(int i=network.size()-1; i>=0; i--){
            NNLayer layer = network.get(i);
            INDArray delta = layer.getDelta(getOutput(input, i-1), curError);
            INDArray weightsT = layer.getWeights().dup().transpose();
            curError = weightsT.mmul(delta);
            if(i== id){
                break;
            }
        }

        return curError;
    }

    public NNLayer getLayer(int id){
        return network.get(id);
    }

    public int size(){
        return network.size();
    }
}
