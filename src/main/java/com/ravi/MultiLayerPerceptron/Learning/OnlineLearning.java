package com.ravi.MultiLayerPerceptron.Learning;

import com.ravi.MultiLayerPerceptron.Error.ErrorFunction;
import com.ravi.MultiLayerPerceptron.MLP.MLPerceptron;
import com.ravi.MultiLayerPerceptron.Training.TrainingAlgorithm;
import com.ravi.MultiLayerPerceptron.Utils.ArrayUtils;
import com.ravi.MultiLayerPerceptron.Utils.Logger;

import java.util.Random;

/**
 * Created by 611445924 on 08/03/2017.
 */
public class OnlineLearning implements LearningAlgorithm {
    private double validationSize;
    private TrainingAlgorithm trainingAlgorithm;
    private ErrorFunction ef;
    private MLPerceptron perceptron;
    private MLPerceptron bestWeights;
    private double bestValidationError = Double.MAX_VALUE;

    public OnlineLearning(double validationSize, TrainingAlgorithm trainingAlgorithm, ErrorFunction ef, MLPerceptron perceptron) {
        this.validationSize = validationSize;
        this.trainingAlgorithm = trainingAlgorithm;
        this.ef = ef;
        this.perceptron = perceptron;
        this.bestWeights = perceptron;
    }

    private double calculateError(double[] input, double[] outputs){
        double[] actOutputs = perceptron.getOutput(input);
        return ef.error(outputs, actOutputs);
    }

    private double localError(double[] desiredOutputs, double[] actOutputs){
        int n=desiredOutputs.length;
        double error = 0.0;
        for(int t=0; t<n; t++){
            error = error + (desiredOutputs[t] - actOutputs[t]);
        }
        return error;
    }

    public double runEpoch(double[][] inputs, double[][] outputs, int trainingSetSize){
        double totalError = 0.0;
        for(int t=0; t<trainingSetSize; t++){
            double[] actOutputs = perceptron.getOutput(inputs[t]);
            double ekt = localError(outputs[t], actOutputs);
            totalError = totalError + calculateError(inputs[t], outputs[t]);

            Logger.debugLog("Running online learning for input "+t);
            StringBuilder msg = new StringBuilder("Inputs are ");
            for(int j=0; j<inputs[t].length; j++){
                msg.append(inputs[t][j] + " ");
            }
            Logger.debugLog(msg.toString());
            Logger.debugLog("Desired Output "+outputs[t][0]);
            Logger.debugLog("Actual Output "+actOutputs[0]);
            Logger.debugLog("Error "+ekt);

            trainingAlgorithm.initialise(perceptron, t);
            perceptron.updateWeightKI(trainingAlgorithm.calculateDeltaWeightKI(inputs[t], ekt, trainingAlgorithm.calculateDeltaK(inputs[t], ekt)));
            perceptron.updateWeightIJ(trainingAlgorithm.calculateDeltaWeightIJ(inputs[t], ekt, trainingAlgorithm.calculateDeltaI(inputs[t], ekt)));


            Logger.debugLog("Finished online learning for input "+t);
            Logger.debugLog("*****************************************************************************************************************");
        }

        return totalError;
    }

    private double validation(double[][] inputs, double[][] outputs, int trainingSetSize){
        double totalError = 0.0;
        for(int t=trainingSetSize; t<inputs.length; t++){
            double[] actOutputs = perceptron.getOutput(inputs[t]);
            totalError = totalError + calculateError(inputs[t], outputs[t]);
        }

        return totalError;
    }

    public void train(double[][] inputs, double[][] outputs) {
        int trainingSetSize = (int) (inputs.length- inputs.length*validationSize);
        int count=0;
        double oldValidationError = Double.POSITIVE_INFINITY;
        int validationErrorsCount = 0;
        while(count < 50000){
            Logger.log("Running Epoch Number "+count);
            double error = runEpoch(inputs, outputs, trainingSetSize);
            Logger.log("Finished Epoch Number "+count);
            if(error < 0.0001){
                break;
            }else{
                Logger.log("Total Error "+error);
            }
            double validationError = validation(inputs, outputs, trainingSetSize);
            Logger.log("Validation Error "+validationError);
            if(oldValidationError >= validationError){
                validationErrorsCount = 0;
            }else{
                validationErrorsCount++;
                if(validationErrorsCount > 1000) {
                    break;
                }
            }
            oldValidationError = validationError;

            if(validationError < bestValidationError){
                bestValidationError = validationError;
                bestWeights.setWeightKI(perceptron.getWeightKI());
                bestWeights.setWeightIJ(perceptron.getWeightIJ());
            }

            Logger.log("##################################################################################################################");
            count++;
        }

        perceptron = bestWeights;

    }
}
