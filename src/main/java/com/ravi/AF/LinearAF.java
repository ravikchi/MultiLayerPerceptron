package com.ravi.AF;

/**
 * Created by 611445924 on 08/03/2017.
 */
public class LinearAF implements ActivationFunction{
    public double activate(double x) {
        return x;
    }

    public double derivative(double x) {
        return 1.0;
    }
}
