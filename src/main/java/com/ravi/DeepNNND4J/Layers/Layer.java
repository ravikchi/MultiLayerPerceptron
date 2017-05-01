package com.ravi.DeepNNND4J.Layers;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Created by 611445924 on 01/05/2017.
 */
public interface Layer {
    public StandardLayer clone();
    public INDArray getOutput(INDArray input);
    public INDArray getDelta(INDArray input, INDArray error);
    public INDArray getWeights();
    public void setWeights(INDArray weights);
    public INDArray getBias();
    public void setBias(INDArray bias);
    public INDArray getOldDeltaWeights();
    public void setOldDeltaWeights(INDArray oldDeltaWeights);
    public INDArray getOldDeltaBias();
    public void setOldDeltaBias(INDArray oldDeltaBias);
}
