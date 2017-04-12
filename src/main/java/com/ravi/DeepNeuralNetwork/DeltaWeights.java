package com.ravi.DeepNeuralNetwork;

/**
 * Created by rc16956 on 12/04/2017.
 */
public class DeltaWeights {
    double[][] weigths;
    double[] bias;

    public DeltaWeights(double[][] deltaWeigths, double[] deltaBias) {
        this.weigths = deltaWeigths;
        this.bias = deltaBias;
    }

    public double[][] getWeigths() {
        return weigths;
    }

    public void setWeigths(double[][] weigths) {
        this.weigths = weigths;
    }

    public double[] getBias() {
        return bias;
    }

    public void setBias(double[] bias) {
        this.bias = bias;
    }
}
