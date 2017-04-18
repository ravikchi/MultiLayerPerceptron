package com.ravi.DeepNNND4J;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Created by rc16956 on 12/04/2017.
 */
public class DeltaWeights {
    INDArray weigths;
    INDArray bias;

    public DeltaWeights(INDArray weigths, INDArray bias) {
        this.weigths = weigths;
        this.bias = bias;
    }

    public INDArray getWeigths() {
        return weigths;
    }

    public void setWeigths(INDArray weigths) {
        this.weigths = weigths;
    }

    public INDArray getBias() {
        return bias;
    }

    public void setBias(INDArray bias) {
        this.bias = bias;
    }
}
