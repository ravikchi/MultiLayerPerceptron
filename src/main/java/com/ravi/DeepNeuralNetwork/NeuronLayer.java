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
    private double bias[];

    public NeuronLayer(ActivationFunction activationFunction, int numberOfInputs, int numberOfNeurons) {
        this.activationFunction = activationFunction;
        weights = new double[numberOfNeurons][numberOfInputs];
        bias = new double[numberOfNeurons];
    }

    public double[] getOutput(double[] input){
        double output[] = MatrixMath.multiply(weights, input);
        return MapFunc.map(activationFunction, MatrixMath.add(output, bias));
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
}
