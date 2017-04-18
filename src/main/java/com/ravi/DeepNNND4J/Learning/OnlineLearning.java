package com.ravi.DeepNNND4J.Learning;

import com.ravi.DeepNNND4J.DeltaWeights;
import com.ravi.DeepNNND4J.Error.ErrorFunction;
import com.ravi.DeepNNND4J.NNLayer;
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

    public OnlineLearning(NNetworkND4j network, TrainingAlgorithm trainingAlgorithm) {
        this.network = network;
        this.trainingAlgorithm = trainingAlgorithm;
    }

    @Override
    public NNetworkND4j train(INDArray inputs, INDArray outputs) {
        int count = 0;
        double error = 0.0;
        while(count<50000){
            Logger.log("Starting Epoch "+count);
            error = epoch(inputs, outputs);
            Logger.log("Finished Epoch "+count);
            Logger.log("Total Error "+error);
            if(error < 0.0000001){
                break;
            }
            count++;
        }

        return network;
    }

    public double epoch(INDArray inputs, INDArray desOutputs){
        double totalError = 0.0;
        ErrorFunction ef = new ErrorFunction();

        for(int i=0; i<inputs.columns(); i++) {
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
        }

        return totalError;
    }
}
