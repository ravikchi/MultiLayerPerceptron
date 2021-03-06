package com.ravi.MultiLayerPerceptron;

import com.ravi.DeepNeuralNetwork.AF.ActivationFunction;
import com.ravi.DeepNeuralNetwork.AF.SigmoidAF;

/**
 * Hello world!
 *
 */
public class App 
{
    public static void main( String[] args )
    {
        ActivationFunction af = new SigmoidAF(1.0);
        System.out.println(af.activate(1));
        System.out.println(Math.exp(1));
        System.out.println(af.derivative(1));
    }
}
