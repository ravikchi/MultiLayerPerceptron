package com.ravi.DeepNeuralNetwork.Training;

import com.ravi.DeepNeuralNetwork.DeltaWeights;
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

    public DeltaWeights train(NeuronLayer layer, double[] input, double[] error) {
        double[] deltaL = layer.getDelta(input, error);

        double[][] oldDeltaWeights = layer.getOldWeights();
        double[][] deltaWeigths = MatrixMath.copy(layer.getWeights());

        double[] oldBias = layer.getOldBias();
        double[] deltaBias = MatrixMath.copy(layer.getBias());
        for (int j = 0; j < deltaL.length; j++) {
            deltaWeigths[j] = MatrixMath.add(MatrixMath.scalarProduct(input, deltaL[j] * learningRate), MatrixMath.scalarProduct(oldDeltaWeights[j], alpha));
            deltaBias[j] = deltaL[j] * learningRate + oldBias[j]*alpha;
        }

        layer.setOldWeights(MatrixMath.copy(deltaWeigths));
        layer.setOldBias(MatrixMath.copy(deltaBias));

        DeltaWeights weights = new DeltaWeights(deltaWeigths, deltaBias);
        return weights;
    }
}
