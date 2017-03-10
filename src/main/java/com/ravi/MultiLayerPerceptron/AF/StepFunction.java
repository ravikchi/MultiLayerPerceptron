package com.ravi.MultiLayerPerceptron.AF;

/**
 * Created by 611445924 on 08/03/2017.
 */
public class StepFunction implements ActivationFunction{
    public double activate(double x) {
        if(x >= 0){
            return 1.0;
        }else{
            return 0.0;
        }
    }

    public double derivative(double x) {
        return 0;
    }
}
