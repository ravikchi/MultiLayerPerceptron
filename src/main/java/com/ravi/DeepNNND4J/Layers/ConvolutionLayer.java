package com.ravi.DeepNNND4J.Layers;

import com.ravi.DeepNNND4J.AF.ActivationFunction;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

/**
 * Created by 611445924 on 01/05/2017.
 */
public class ConvolutionLayer extends StandardLayer {

    private int m;
    private int n;
    private int i;
    private int j;

    public ConvolutionLayer(ActivationFunction activationFunction, int numberOfInputs, int numberOfNeurons) {
        super(activationFunction, numberOfInputs, numberOfNeurons);
    }

    public ConvolutionLayer(ActivationFunction activationFunction, int numberOfNeurons, int m, int n, int i, int j){
        super(activationFunction);
        super.weights = Nd4j.rand(m, n);
        super.bias = Nd4j.rand(numberOfNeurons, 1);
        this.m = m;
        this.n = n;
        this.i = i;
        this.j = j;
    }

    private INDArray crossCorrelation(INDArray input){
        INDArray output = input.dup();
        for(int i=0; i<this.i; i++){
            for(int j=0; j<this.j; j++){
                INDArray temp = weights.mmul(input.get(NDArrayIndex.interval(i+m, j+n)));
                output.get(NDArrayIndex.interval(i+m, j+n)).assign(temp);
            }
        }
        return output;
    }

    public INDArray getOutput(INDArray input){
        INDArray output = crossCorrelation(input);

        return activationFunction.activate(output);
    }

    public INDArray getDelta(INDArray input, INDArray error){
        INDArray output = crossCorrelation(input);

        output = activationFunction.derivative(output);
        //return output.transpose().mmul(error);
        return product(output, error);
    }

}
