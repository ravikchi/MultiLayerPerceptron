package com.ravi.DeepNNND4J.Examples;

import com.ravi.DeepNNND4J.AF.StepFunction;
import com.ravi.DeepNNND4J.NNLayer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Created by 611445924 on 18/04/2017.
 */
public class XORLearning {
    public static void main(String[] args){
        XORLearning xorLearning = new XORLearning();
        INDArray inputs = Nd4j.create(new double[]{1,0,1,0,0,1,1,0}, new int[]{2,4}, 'c');

        for(int i=0; i<inputs.columns(); i++){
            INDArray input = inputs.getColumn(i);
            System.out.println(input);
            //INDArray trainingOutput = xorLearning.getTrainingOutput(trainingInput);

            //System.out.println(trainingOutput);

            NNLayer layer1 = new NNLayer(new StepFunction(), 2,2);
            layer1.setWeights(Nd4j.create(new double[]{1,1,1,1}, new int[]{2,2}));
            layer1.setBias(Nd4j.create(new double[]{-1.5, -0.5}, new int[]{2,1}));

            INDArray outputH = layer1.getOutput(input);

            NNLayer layer2 = new NNLayer(new StepFunction(), 1, 2);
            layer2.setWeights(Nd4j.create(new double[]{-2, 1}, new int[]{1,2}));
            layer2.setBias(Nd4j.create(new double[]{-0.5}, new int[]{1,1}));

            INDArray outputY = layer2.getOutput(outputH);

            System.out.println(outputY);
        }


    }

    private INDArray stepFunction(INDArray output){
        for(int i=0; i<output.linearView().length(); i++){
            if(output.linearView().getDouble(i) >= 0){
                output.linearView().putScalar(i, 1.0);
            }else{
                output.linearView().putScalar(i, 0.0);
            }
        }
        return output;
    }

    public INDArray getOutput(INDArray input){
        INDArray weightsH = Nd4j.create(new double[]{1,1,1,1}, new int[]{2,2});
        INDArray biasH = Nd4j.create(new double[]{-1.5, -0.5}, new int[]{2,1});

        INDArray outputH = weightsH.mmul(input);
        outputH.addi(biasH);

        outputH = stepFunction(outputH);

        INDArray weightsY = Nd4j.create(new double[]{-2, 1}, new int[]{1,2});
        INDArray biasY = Nd4j.create(new double[]{-0.5}, new int[]{1,1});

        INDArray outputY = weightsY.mmul(outputH);
        outputY.addi(biasY);

        outputY = stepFunction(outputY);

        return outputY;
    }
}
