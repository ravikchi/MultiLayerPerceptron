package com.ravi.DeepNeuralNetwork.AF;

/**
 * Created by 611445924 on 06/03/2017.
 */
public interface ActivationFunction {
    public double activate(double x);
    public double derivative(double x);
}
