package com.ravi.MultiLayerPerceptron.Learning;

import com.ravi.MultiLayerPerceptron.Error.ErrorFunction;
import com.ravi.MultiLayerPerceptron.MLP.MLPerceptron;
import com.ravi.MultiLayerPerceptron.Training.TrainingAlgorithm;
import com.ravi.MultiLayerPerceptron.Utils.Logger;

/**
 * Created by rc16956 on 10/03/2017.
 */
public class BatchLearning implements LearningAlgorithm {
    private double validationSize;
    private TrainingAlgorithm trainingAlgorithm;
    private ErrorFunction ef;
    private MLPerceptron perceptron;

    public BatchLearning(double validationSize, TrainingAlgorithm trainingAlgorithm, ErrorFunction ef, MLPerceptron perceptron) {
        this.validationSize = validationSize;
        this.trainingAlgorithm = trainingAlgorithm;
        this.ef = ef;
        this.perceptron = perceptron;
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
        double[][] deltaWeightsKI = new double[perceptron.getNumberOfOutputs()][perceptron.getNumberOfHiddenNeurons()+1];
        double[][] deltaWeightsIJ = new double[perceptron.getNumberOfHiddenNeurons()][perceptron.getNumberOfInputs()+1];

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
            double[][] localDeltaWeightsKI = trainingAlgorithm.calculateDeltaWeightKI(inputs[t], ekt, trainingAlgorithm.calculateDeltaK(inputs[t], ekt));
            deltaWeightsKI = updateDeltaWeights(localDeltaWeightsKI, deltaWeightsKI);

            double[][] localDeltaWeightsIJ = trainingAlgorithm.calculateDeltaWeightIJ(inputs[t], ekt, trainingAlgorithm.calculateDeltaI(inputs[t], ekt));
            deltaWeightsIJ = updateDeltaWeights(localDeltaWeightsIJ, deltaWeightsIJ);


            Logger.debugLog("Finished online learning for input "+t);
            Logger.debugLog("*****************************************************************************************************************");
        }

        perceptron.updateWeightKI(deltaWeightsKI);

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

            double[][] localDeltaWeightsIJ = trainingAlgorithm.calculateDeltaWeightIJ(inputs[t], ekt, trainingAlgorithm.calculateDeltaI(inputs[t], ekt));
            deltaWeightsIJ = updateDeltaWeights(localDeltaWeightsIJ, deltaWeightsIJ);


            Logger.debugLog("Finished online learning for input "+t);
            Logger.debugLog("*****************************************************************************************************************");
        }
        perceptron.updateWeightIJ(deltaWeightsIJ);

        return totalError;
    }

    private double[][] updateDeltaWeights(double[][] localDeltaWeights, double[][] deltaWeights){
        for(int k=0; k<localDeltaWeights.length; k++){
            for(int i=0; i<localDeltaWeights[k].length; i++){
                deltaWeights[k][i] = deltaWeights[k][i] = localDeltaWeights[k][i];
            }
        }
        return deltaWeights;
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
            Logger.log("##################################################################################################################");
            count++;
        }

    }
}
