package com.ravi.DeepNNND4J.Training;

import com.ravi.DeepNNND4J.DeltaWeights;
import com.ravi.DeepNNND4J.NNLayer;
import com.ravi.Utils.Logger;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Created by 611445924 on 18/04/2017.
 */
public class BackPropagation implements TrainingAlgorithm {
    private double learningRate = 0.1;
    private double alpha = 0.01;

    @Override
    public DeltaWeights train(NNLayer layer, INDArray input, INDArray error) {
        INDArray deltaY = layer.getDelta(input, error);
        Logger.debugLog(deltaY.toString());

        INDArray oldDeltaWeights = layer.getOldDeltaWeights();
        oldDeltaWeights = oldDeltaWeights.mul(alpha);

        INDArray oldDeltaBias = layer.getOldDeltaBias();
        oldDeltaBias = oldDeltaBias.mul(alpha);

        INDArray deltaWeightsY = layer.getWeights().dup();
        INDArray deltaBiasY = layer.getBias().dup();

        deltaY = deltaY.mul(learningRate);

        for(int j=0; j<deltaY.rows(); j++){
            deltaWeightsY.putRow(j, input.transpose().mul(deltaY.getRow(j)).add(oldDeltaWeights.getRow(j)));
            deltaBiasY.putRow(j, deltaY.getRow(j).add(oldDeltaBias.getRow(j)));
        }

        layer.setOldDeltaBias(deltaWeightsY.dup());
        layer.setOldDeltaBias(deltaBiasY.dup());

        DeltaWeights weights = new DeltaWeights(deltaWeightsY, deltaBiasY);
        return weights;
    }
}
