package com.ravi.DeepNNND4J.Layers;

import com.ravi.DeepNNND4J.AF.ActivationFunction;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Created by 611445924 on 01/05/2017.
 */
public class ConvolutionLayer extends StandardLayer {

    private int m;
    private int n;

    public ConvolutionLayer(ActivationFunction activationFunction, int numberOfInputs, int numberOfNeurons) {
        super(activationFunction, numberOfInputs, numberOfNeurons);
    }

    public ConvolutionLayer(ActivationFunction activationFunction, int numberOfInputs, int numberOfNeurons, int m, int n){
        super(activationFunction, numberOfInputs, numberOfNeurons);
        this.m = m;
        this.n = n;
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
        //return output.transpose().mmul(error);
        return product(output, error);
    }

}
