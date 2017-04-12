package com.ravi.DeepNeuralNetwork.Learning;

import com.ravi.DeepNeuralNetwork.DeltaWeights;
import com.ravi.DeepNeuralNetwork.NeuralNetwork;
import com.ravi.DeepNeuralNetwork.NeuronLayer;
import com.ravi.DeepNeuralNetwork.Training.BackPropagation;
import com.ravi.DeepNeuralNetwork.Training.TrainingAlgorithm;
import com.ravi.Error.ErrorFunction;
import com.ravi.Error.RMSE;
import com.ravi.Utils.Logger;
import com.ravi.Utils.MatrixMath;

/**
 * Created by rc16956 on 12/04/2017.
 */
public class OnlineLearning implements LearningAlgorithm {
    TrainingAlgorithm trainingAlgo;
    NeuralNetwork network;
    private ErrorFunction ef = new RMSE();

    public double getValidationSize() {
        return validationSize;
    }

    public void setValidationSize(double validationSize) {
        this.validationSize = validationSize;
    }

    private double validationSize =0.0;
    private NeuralNetwork bestWeights;
    private double bestValidationError = Double.MAX_VALUE;

    public OnlineLearning(BackPropagation trainingAlgo, NeuralNetwork network) {
        this.trainingAlgo = trainingAlgo;
        this.network = network;
        this.bestWeights = network;
    }

    public OnlineLearning(TrainingAlgorithm trainingAlgo, NeuralNetwork network, double validationSize) {
        this.trainingAlgo = trainingAlgo;
        this.network = network;
        this.validationSize = validationSize;
    }

    public double epoch(double[][] inputs, double[][] outputs, int trainingSetSize){
        double totalError = 0.0;
        for(int t=0; t<trainingSetSize; t++){
            double[] input = inputs[t];
            double[] actOutput = network.getOutput(input);

            double[] error = MatrixMath.substract(outputs[t], actOutput);
            totalError = totalError + ef.error(outputs[t], actOutput);

            Logger.debugLog("Running online learning for input "+t);
            StringBuilder msg = new StringBuilder("Inputs are ");
            for(int j=0; j<inputs[t].length; j++){
                msg.append(inputs[t][j] + " ");
            }
            Logger.debugLog(msg.toString());
            Logger.debugLog("Desired Output "+outputs[t][0]);
            Logger.debugLog("Actual Output "+actOutput[0]);
            Logger.debugLog("Error "+ef.error(outputs[t], actOutput));

            for(int i=network.size()-1; i>=0; i--){
                NeuronLayer layer = network.getLayer(i);

                //get the input to the layer
                double[] localInput = network.getOutput(input, i-1);
                double[] localError = network.getError(network.getOutput(input, i), error, i+1);

                //Calculate the Delta Weigths
                DeltaWeights deltaWeights = trainingAlgo.train(layer, localInput, localError);

                //Update the Weights
                logWeights(layer.getWeights(), "Old Weights");
                layer.setWeights(MatrixMath.add(layer.getWeights(), deltaWeights.getWeigths()));
                logWeights(layer.getWeights(), "New Weights");
                layer.setBias(MatrixMath.add(layer.getBias(), deltaWeights.getBias()));
            }

            Logger.debugLog("Finished online learning for input "+t);
            Logger.debugLog("*****************************************************************************************************************");
        }

        Logger.log("Total Error for epoch "+totalError);
        return totalError;
    }

    private double validation(double[][] inputs, double[][] outputs, int trainingSetSize){
        double totalError = 0.0;
        for(int t=0; t<trainingSetSize; t++) {
            double[] input = inputs[t];
            double[] actOutput = network.getOutput(input);

            double[] error = MatrixMath.substract(outputs[t], actOutput);
            totalError = totalError + ef.error(outputs[t], actOutput);
        }

        return totalError;
    }



    private void logWeights(double[][] weights, String str){
        Logger.debugLog(str);
        for(int i=0; i<weights.length; i++){
            StringBuilder msg = new StringBuilder();
            for(int j=0; j<weights[i].length; j++){
                msg.append(weights[i][j]);
                msg.append("    ");
            }
            Logger.debugLog(msg.toString());
        }
    }

    public NeuralNetwork train(double[][] inputs, double[][] outputs) {
        int trainingSetSize = (int) (inputs.length- inputs.length*validationSize);
        double oldValidationError = Double.POSITIVE_INFINITY;
        int validationErrorsCount = 0;
        for(int t=0; t<50000; t++) {
            Logger.log("Running Epoch Number "+t);
            double error = epoch(inputs, outputs, trainingSetSize);
            if(error < 0.000001){
                break;
            }
            double validationError = validation(inputs, outputs, trainingSetSize);
            Logger.log("Validation Error "+validationError);
            if(oldValidationError >= validationError){
                validationErrorsCount = 0;
            }else{
                validationErrorsCount++;
                if(validationErrorsCount > 10000) {
                    break;
                }
            }
            oldValidationError = validationError;

            if(validationError < bestValidationError){
                bestValidationError = validationError;
                bestWeights = network.clone();
            }

            Logger.log("Finished Epoch Number "+t);
            Logger.log("##################################################################################################################");
        }

        return bestWeights;
    }
}
