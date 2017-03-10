package com.ravi.MultiLayerPerceptron.Training;

import com.ravi.MultiLayerPerceptron.MLP.MLPerceptron;

/**
 * Created by 611445924 on 08/03/2017.
 */
public interface TrainingAlgorithm {
    public void initialise(MLPerceptron perceptron, int t);
    public double[][] calculateDeltaWeightIJ(double[] inputs, double ekt, double[] deltaI);
    public double[][] calculateDeltaWeightKI(double[] inputs, double ekt, double[] deltaK);
    public double[] calculateDeltaK(double[] inputs, double ekt);
    public double[] calculateDeltaI(double[] inputs, double ekt);
}
