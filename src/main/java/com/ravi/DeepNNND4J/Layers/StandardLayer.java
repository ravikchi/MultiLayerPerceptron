package com.ravi.DeepNNND4J.Layers;

import com.ravi.DeepNNND4J.AF.ActivationFunction;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Created by 611445924 on 18/04/2017.
 */
public class StandardLayer implements Layer {
    protected ActivationFunction activationFunction;
    protected INDArray weights;
    protected INDArray bias;

    protected INDArray oldDeltaWeights;
    protected INDArray oldDeltaBias;


    public StandardLayer(ActivationFunction activationFunction, int numberOfInputs, int numberOfNeurons) {
        this.activationFunction = activationFunction;
        this.weights = Nd4j.rand(numberOfNeurons, numberOfInputs);
        this.bias = Nd4j.rand(numberOfNeurons, 1);

        this.oldDeltaWeights = Nd4j.zeros(numberOfNeurons, numberOfInputs);
        this.oldDeltaBias = Nd4j.zeros(numberOfNeurons, 1);
    }

    public StandardLayer clone(){
        StandardLayer layer = new StandardLayer(activationFunction, weights.columns(), weights.rows());
        layer.setWeights(this.weights.dup());
        layer.setBias(this.bias.dup());

        layer.setOldDeltaWeights(this.oldDeltaWeights.dup());
        layer.setOldDeltaBias(this.oldDeltaBias.dup());

        return layer;
    }

    public INDArray getOutput(INDArray input){
        INDArray output = weights.mmul(input);
        output.addi(bias);

        return activationFunction.activate(output);
    }

    public INDArray getDelta(INDArray input, INDArray error){
        INDArray output = weights.mmul(input);
        output.addi(bias);

        output = activationFunction.derivative(output);
        //return output.transpose().mmul(error);
        return product(output, error);
    }

    protected INDArray product(INDArray a, INDArray b){
        INDArray c = Nd4j.zeros(a.rows(), a.columns());
        for(int i=0; i<a.linearView().length(); i++){
            c.linearView().putScalar(i, a.linearView().getDouble(i) * b.linearView().getDouble(i));
        }

        return c;
    }

    public INDArray getWeights() {
        return weights;
    }

    public void setWeights(INDArray weights) {
        this.weights = weights;
    }

    public INDArray getBias() {
        return bias;
    }

    public void setBias(INDArray bias) {
        this.bias = bias;
    }

    public INDArray getOldDeltaWeights() {
        return oldDeltaWeights;
    }

    public void setOldDeltaWeights(INDArray oldDeltaWeights) {
        this.oldDeltaWeights = oldDeltaWeights;
    }

    public INDArray getOldDeltaBias() {
        return oldDeltaBias;
    }

    public void setOldDeltaBias(INDArray oldDeltaBias) {
        this.oldDeltaBias = oldDeltaBias;
    }
}
