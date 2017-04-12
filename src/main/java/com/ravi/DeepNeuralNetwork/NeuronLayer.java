package com.ravi.DeepNeuralNetwork;

import com.ravi.AF.ActivationFunction;
import com.ravi.Utils.MapFunc;
import com.ravi.Utils.MatrixMath;

/**
 * Created by 611445924 on 10/04/2017.
 */
public class NeuronLayer {
    private ActivationFunction activationFunction;
    private double weights[][];
    private double oldWeights[][];
    private double bias[];
    private double oldBias[];

    public NeuronLayer(ActivationFunction activationFunction, int numberOfInputs, int numberOfNeurons) {
        this.activationFunction = activationFunction;
        weights = new double[numberOfNeurons][numberOfInputs];
        oldWeights = new double[numberOfNeurons][numberOfInputs];
        bias = new double[numberOfNeurons];
        oldBias = new double[numberOfNeurons];
        initialise();
    }

    public NeuronLayer clone(){
        NeuronLayer newLayer = new NeuronLayer(this.activationFunction, weights[0].length, weights.length);
        newLayer.setWeights(weights);
        newLayer.setBias(bias);
        return newLayer;
    }

    private void initialise(){
        for(int i=0; i<weights.length; i++){
            for(int j=0; j<weights[i].length; j++){
                weights[i][j] = Math.random();
            }
            bias[i] = Math.random();
        }
    }

    public double[] getOutput(double[] input){
        double output[] = MatrixMath.multiply(weights, input);
        return MapFunc.map(activationFunction, MatrixMath.add(output, bias), true);
    }

    public double[] getDelta(double[] input, double[] error){
        double output[] = MatrixMath.multiply(weights, input);
        output = MapFunc.map(activationFunction, MatrixMath.add(output, bias), false);//get the derivative of the inputs

        return MatrixMath.arrayProduct(output, error); //multiply the derivative of the inputs with the error
    }

    public double[][] getWeights() {
        return weights;
    }

    public void setWeights(double[][] weights) {
        this.weights = weights;
    }

    public double[] getBias() {
        return bias;
    }

    public void setBias(double[] bias) {
        this.bias = bias;
    }

    public double[][] getOldWeights() {
        return oldWeights;
    }

    public void setOldWeights(double[][] oldWeights) {
        this.oldWeights = oldWeights;
    }

    public double[] getOldBias() {
        return oldBias;
    }

    public void setOldBias(double[] oldBias) {
        this.oldBias = oldBias;
    }
}
