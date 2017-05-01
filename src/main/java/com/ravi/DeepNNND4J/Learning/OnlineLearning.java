package com.ravi.DeepNNND4J.Learning;

import com.ravi.DeepNNND4J.DeltaWeights;
import com.ravi.DeepNNND4J.Error.ErrorFunction;
import com.ravi.DeepNNND4J.Layers.NNLayer;
import com.ravi.DeepNNND4J.NNetworkND4j;
import com.ravi.DeepNNND4J.Training.TrainingAlgorithm;
import com.ravi.Utils.Logger;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Created by 611445924 on 18/04/2017.
 */
public class OnlineLearning implements LearningAlgorithm{
    NNetworkND4j network;
    TrainingAlgorithm trainingAlgorithm;
    NNetworkND4j bestNetwork;
    EarlyStopCriteria earlyStopCriteria = new EarlyStopCriteria(0.0, 50000, 0.000001, 10000);

    public OnlineLearning(NNetworkND4j network, TrainingAlgorithm trainingAlgorithm) {
        this.network = network;
        this.trainingAlgorithm = trainingAlgorithm;
    }

    private double validation(INDArray inputs, INDArray desOutputs){
        double totalError = 0.0;
        ErrorFunction ef = new ErrorFunction();

        int n=0;

        for(int i=validationSize(inputs); i<inputs.columns(); i++) {
            INDArray input = inputs.getColumn(i);
            INDArray desOutput = desOutputs.getColumn(i);

            INDArray output = network.getOutput(input);
            Logger.debugLog("Actual Output :" + output);

            INDArray error = desOutput.sub(output);
            Logger.debugLog("Local Error " + error);
            totalError = totalError + ef.getError(desOutput, output);

            n++;
        }

        return totalError/n;
    }


    @Override
    public NNetworkND4j train(INDArray inputs, INDArray outputs) {
        int count = 0;
        double error = 0.0;
        double validationError = 0.0;

        int validationCount = 0;

        double oldValidationError = Double.MAX_VALUE;
        while(count<earlyStopCriteria.getMaxEpoch()){
            Logger.log("Starting Epoch "+count);
            error = epoch(inputs, outputs);
            Logger.log("Finished Epoch "+count);

            validationError = validation(inputs, outputs);
            if(validationError < oldValidationError){
                bestNetwork = network.getClone();
                oldValidationError = validationError;
            }else{
                if(validationCount > earlyStopCriteria.getMaxValidationCount()){
                    break;
                }else{
                    validationCount++;
                }
            }

            Logger.log("Total Error "+error);
            Logger.log("Validation Error "+validationError);
            if(error < earlyStopCriteria.getMinError()){
                break;
            }
            count++;
        }

        return network;
    }

    @Override
    public void setEarlyStopCriteria(EarlyStopCriteria earlyStopCriteria) {
        this.earlyStopCriteria = earlyStopCriteria;

    }

    private int validationSize(INDArray inputs){
        int size = (int) (inputs.columns() - (inputs.columns() * earlyStopCriteria.getValidationSize()));
        return size;
    }

    public double epoch(INDArray inputs, INDArray desOutputs){
        double totalError = 0.0;
        ErrorFunction ef = new ErrorFunction();

        int n=0;

        for(int i=0; i<validationSize(inputs); i++) {
            INDArray input = inputs.getColumn(i);
            INDArray desOutput = desOutputs.getColumn(i);

            INDArray output = network.getOutput(input);
            Logger.debugLog("Actual Output :"+ output);

            INDArray error = desOutput.sub(output);
            Logger.debugLog("Local Error "+ error);
            totalError = totalError + ef.getError(desOutput, output);

            for(int j=network.size()-1; j>=0; j--){
                INDArray localInput = network.getOutput(input, j-1);
                INDArray localError = network.getError(input, error, j+1);

                NNLayer localLayer = network.getLayer(j);
                DeltaWeights weights = trainingAlgorithm.train(localLayer, localInput, localError);

                Logger.debugLog("Original Layer1 Weights");
                Logger.debugLog(localLayer.getWeights().toString());
                localLayer.setWeights(localLayer.getWeights().add(weights.getWeigths()));
                Logger.debugLog("Altered Layer1 Weights");
                Logger.debugLog(localLayer.getWeights().toString());
                localLayer.setBias(localLayer.getBias().add(weights.getBias()));

                Logger.debugLog("***************************************************************");
            }
            n++;
        }

        return totalError/n;
    }
}
