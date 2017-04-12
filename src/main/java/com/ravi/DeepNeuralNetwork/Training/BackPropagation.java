package com.ravi.DeepNeuralNetwork.Training;

import com.ravi.DeepNeuralNetwork.NeuronLayer;
import com.ravi.Utils.MatrixMath;

/**
 * Created by rc16956 on 10/04/2017.
 */
public class BackPropagation implements TrainingAlgorithm {
    private double learningRate;
    private double alpha;


    public BackPropagation(double learningRate, double alpha) {
        this.learningRate = learningRate;
        this.alpha = alpha;
    }

    public double[][] train(NeuronLayer layer, double[] input, double[] error) {
        double[] deltaL = layer.getDelta(input, error);

        double[][] oldDeltaWeights = layer.getOldWeights();
        double[][] deltaWeigths = layer.getWeights();
        for (int j = 0; j < deltaL.length; j++) {
            deltaWeigths[j] = MatrixMath.add(MatrixMath.scalarProduct(input, deltaL[j] * learningRate), MatrixMath.scalarProduct(oldDeltaWeights[j], alpha));
        }

        layer.setOldWeights(MatrixMath.copy(deltaWeigths));
        return deltaWeigths;
    }
}
