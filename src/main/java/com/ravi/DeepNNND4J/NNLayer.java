package com.ravi.DeepNNND4J;

import com.ravi.DeepNNND4J.AF.ActivationFunction;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Created by 611445924 on 18/04/2017.
 */
public class NNLayer {
    private INDArray weights;
    private INDArray bias;
    private ActivationFunction activationFunction;

    public NNLayer(ActivationFunction activationFunction, int numberOfInputs, int numberOfNeurons) {
        this.activationFunction = activationFunction;
        this.weights = Nd4j.rand(numberOfNeurons, numberOfInputs);
        this.bias = Nd4j.rand(numberOfNeurons, 1);
    }

    public INDArray getOutput(INDArray input){
        INDArray output = weights.mmul(input);
        output.addi(bias);

        return activationFunction.activate(output);
    }

    public INDArray getDelta(INDArray input, INDArray error){
        INDArray output = weights.mmul(input);
        output.addi(bias);

        output = activationFunction.derivative(output);
        return output.transpose().mmul(error);
    }

    public INDArray getWeights() {
        return weights;
    }

    public void setWeights(INDArray weights) {
        this.weights = weights;
    }

    public INDArray getBias() {
        return bias;
    }

    public void setBias(INDArray bias) {
        this.bias = bias;
    }
}
