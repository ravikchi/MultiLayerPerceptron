package com.ravi.DeepNNND4J;

import com.ravi.DeepNNND4J.AF.LinearAF;
import com.ravi.DeepNNND4J.AF.SigmoidAF;
import com.ravi.DeepNNND4J.Error.ErrorFunction;
import com.ravi.Utils.Logger;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.conditions.GreaterThanOrEqual;
import org.nd4j.linalg.indexing.conditions.LessThan;

/**
 * Created by 611445924 on 18/04/2017.
 */
public class XORDNNLearning {
    public static void main(String[] args){
        INDArray inputs = Nd4j.create(new double[]{1,0,1,0,0,1,1,0}, new int[]{2,4}, 'c');
        INDArray desOutputs = Nd4j.create(new double[]{1, 1, 0, 0}, new int[]{1, 4});

        NNLayer layer1 = new NNLayer(new SigmoidAF(), 2,2);
        NNLayer layer2 = new NNLayer(new LinearAF(), 2, 1);

        int count=0;
        double error = 0.0;
        while(count<50000){
            error = epoch(inputs, desOutputs, layer1, layer2);
            Logger.log("Total Error "+error);
            if(error < 0.01){
                break;
            }
            count++;
        }

        Logger.debugLog("###############################################################");
        Logger.debugLog("Finished Training");

        for(int i=0; i<inputs.columns(); i++) {
            INDArray input = inputs.getColumn(i);
            INDArray desOutput = desOutputs.getColumn(i);

            INDArray outputH = layer1.getOutput(input);
            INDArray outputY = layer2.getOutput(outputH);
            Logger.debugLog("Output :"+ outputY);
        }

    }

    public static double epoch(INDArray inputs, INDArray desOutputs, NNLayer layer1, NNLayer layer2){
        double totalError = 0.0;
        ErrorFunction ef = new ErrorFunction();

        for(int i=0; i<inputs.columns(); i++) {
            INDArray input = inputs.getColumn(i);
            INDArray desOutput = desOutputs.getColumn(i);

            INDArray outputH = layer1.getOutput(input);
            INDArray outputY = layer2.getOutput(outputH);
            Logger.debugLog("Actual Output :"+ outputY);

            INDArray error = desOutput.sub(outputY);
            Logger.debugLog("Local Error "+ error);

            totalError = totalError + ef.getError(desOutput, outputY);

            Logger.debugLog("Original Layer2 Weights");
            Logger.debugLog(layer2.getWeights().toString());
            updateWeights(layer2, outputH, error);
            Logger.debugLog("Altered Layer2 Weights");
            Logger.debugLog(layer2.getWeights().toString());
            Logger.debugLog("---------------------------------------------------------------");

            outputY = layer2.getOutput(outputH);
            error = desOutput.sub(outputY);

            INDArray deltaY = layer2.getDelta(outputH, error);

            Logger.debugLog("Original Layer1 Weights");
            Logger.debugLog(layer1.getWeights().toString());
            updateWeights(layer1, input, layer2.getWeights().transpose().mmul(deltaY));
            Logger.debugLog("Altered Layer1 Weights");
            Logger.debugLog(layer1.getWeights().toString());
            Logger.debugLog("***************************************************************");
        }

        return totalError;
    }

    public static void updateWeights(NNLayer layer, INDArray input, INDArray error){
        INDArray deltaY = layer.getDelta(input, error);
        Logger.debugLog(deltaY.toString());

        INDArray deltaWeightsY = layer.getWeights().dup();
        INDArray deltaBiasY = layer.getBias().dup();

        deltaY = deltaY.mul(0.1);

        for(int j=0; j<deltaWeightsY.rows(); j++){
            INDArray temp = input.mul(deltaY);
            deltaWeightsY.putRow(j, input.transpose().mul(deltaY));
            deltaBiasY.putRow(j, deltaY);
        }

        layer.setWeights(layer.getWeights().add(deltaWeightsY));
        layer.setBias(layer.getBias().add(deltaBiasY));
    }
}
