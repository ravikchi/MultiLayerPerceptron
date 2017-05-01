package com.ravi.DeepNNND4J.Layers;

import com.ravi.DeepNNND4J.AF.ActivationFunction;

/**
 * Created by 611445924 on 01/05/2017.
 */
public class RBMLayer extends NNLayer {
    public RBMLayer(ActivationFunction activationFunction, int numberOfInputs, int numberOfNeurons) {
        super(activationFunction, numberOfInputs, numberOfNeurons);
    }
}
