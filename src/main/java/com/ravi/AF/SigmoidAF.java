package com.ravi.AF;

/**
 * Created by 611445924 on 08/03/2017.
 */
public class SigmoidAF implements ActivationFunction{
    private double lambda;

    public SigmoidAF(double lambda) {
        this.lambda = lambda;
    }

    public double activate(double x) {
        double sigma = 1/(1+Math.exp(-x*lambda));
        return sigma;
    }

    public double derivative(double x) {
        double sigma = activate(x);
        return (sigma * (1-sigma))*lambda;
    }
}
