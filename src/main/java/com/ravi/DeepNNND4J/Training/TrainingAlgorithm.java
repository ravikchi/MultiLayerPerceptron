package com.ravi.DeepNNND4J.Training;

import com.ravi.DeepNNND4J.Layers.Layer;
import com.ravi.Utils.DeltaWeights;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Created by rc16956 on 12/04/2017.
 */
public interface TrainingAlgorithm {
    public DeltaWeights train(Layer layer, INDArray input, INDArray error);
}
