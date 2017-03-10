package com.ravi.MultiLayerPerceptron.MLP;

import com.ravi.MultiLayerPerceptron.AF.ActivationFunction;
import com.ravi.MultiLayerPerceptron.Utils.Logger;

/**
 * Created by 611445924 on 06/03/2017.
 */
public class MLPerceptron {
    ActivationFunction outputLayerAF;
    ActivationFunction hiddenLayerAF;
    int numberOfHiddenNeurons;
    int numberOfInputs;
    int numberOfOutputs;
    double bias=-1.0;

    double[][] weightKI;
    double[][] weightIJ;

    double[] hi;
    double[] ki;

    public MLPerceptron(ActivationFunction outputLayerAF, ActivationFunction hiddenLayerAF, int numberOfHiddenNeurons, int numberOfInputs, int numberOfOutputs) {
        this.outputLayerAF = outputLayerAF;
        this.hiddenLayerAF = hiddenLayerAF;
        this.numberOfHiddenNeurons = numberOfHiddenNeurons;
        this.numberOfInputs = numberOfInputs;
        this.numberOfOutputs = numberOfOutputs;
        this.weightIJ = new double[numberOfHiddenNeurons][numberOfInputs+1];
        this.weightKI = new double[numberOfOutputs][numberOfHiddenNeurons+1];
        this.hi = new double[numberOfHiddenNeurons];
        this.ki = new double[numberOfOutputs];
        this.initialiseWeights();
    }

    private void initialiseWeights(){
        for(int k=0; k<numberOfOutputs; k++){
            for(int i=0; i<numberOfHiddenNeurons+1; i++) {
                weightKI[k][i] = Math.random();
                Logger.debugLog("Initial Weight of Output neuron  "+k+1+" and hidden neuron "+i+ " is : "+weightKI[k][i]);
            }
        }

        for(int i=0; i<numberOfHiddenNeurons; i++){
            for(int j=0; j<numberOfInputs+1; j++){
                weightIJ[i][j] = Math.random();
                Logger.debugLog("Initial Weight of hidden neuron  "+i+1+" and input "+j+ " is : "+weightIJ[i][j]);
            }
        }

        Logger.debugLog("##################################################################################################################");
    }

    public double[][] getWeightKI() {
        return weightKI;
    }

    public void setWeightKI(double[][] weightKI) {
        this.weightKI = weightKI;
        for(int k=0; k<weightKI.length; k++){
            for(int i=0; i<weightKI[k].length; i++){
                Logger.debugLog("Weight of Output neuron "+k+1+" and hidden neuron "+i+ " is : "+weightKI[k][i]);
            }
        }
    }

    public void updateWeightKI(double[][] deltaWeightKI){
        for(int k=0; k<getNumberOfOutputs(); k++){
            for(int i=0; i<getNumberOfHiddenNeurons()+1; i++) {
                weightKI[k][i] = weightKI[k][i] + deltaWeightKI[k][i];
                Logger.debugLog("Weight of Output neuron "+k+1+" and hidden neuron "+i+ " is : "+weightKI[k][i]);
            }
        }
    }

    public double[][] getWeightIJ() {
        return weightIJ;
    }

    public void setWeightIJ(double[][] weightIJ) {
        this.weightIJ = weightIJ;
        for(int i=0; i<weightIJ.length; i++){
            for(int j=0; j<weightIJ[i].length; j++){
                Logger.debugLog("Weight of hidden neuron  "+i+1+" and input "+j+ " is : "+weightIJ[i][j]);
            }
        }
    }

    public void updateWeightIJ(double[][] deltaweightIJ){
        for(int i=0; i<getNumberOfHiddenNeurons(); i++){
            for(int j=0; j<getNumberOfInputs()+1; j++){
                weightIJ[i][j] = weightIJ[i][j] + deltaweightIJ[i][j];
                Logger.debugLog("Weight of hidden neuron  "+i+1+" and input "+j+ " is : "+weightIJ[i][j]);
            }
        }
    }

    public double[] getOutput(double[] inputs){
        double[] outputs = new double[numberOfOutputs];
        this.hi = new double[numberOfHiddenNeurons];
        this.ki = new double[numberOfOutputs];

        for(int k=0; k<numberOfOutputs; k++){
            outputs[k] = getKi(inputs, k);
        }

        return outputs;
    }

    public double getKi(double[] inputs, int k){

        double output = weightKI[k][0] *bias;
        for (int i = 1; i < weightKI[k].length; i++) {
            output = output + weightKI[k][i] * getHi(inputs, i - 1);
        }
        ki[k] = output;

        return outputLayerAF.activate(output);
    }

    public double getHi(double[] inputs, int i){
        double hi = weightIJ[i][0] * bias;
        for (int j = 1; j < weightIJ[i].length; j++) {
            hi = hi + weightIJ[i][j] * inputs[j - 1];
        }

        this.hi[i] = hi;
        return hiddenLayerAF.activate(hi);
    }

    public double getKiPrime(double[] inputs, int k){
        double output = weightKI[k][0] * bias;
        for (int i = 1; i < weightKI[k].length; i++) {
            output = output + weightKI[k][i] * getHi(inputs, i - 1);
        }
        ki[k] = output;
        return outputLayerAF.derivative(ki[k]);
    }

    public double getHiPrime(double[] inputs, int i){
        double hi = weightIJ[i][0] * bias;
        for (int j = 1; j < weightIJ[i].length; j++) {
            hi = hi + weightIJ[i][j] * inputs[j - 1];
        }

        this.hi[i] = hi;
        return hiddenLayerAF.derivative(this.hi[i]);
    }

    public int getNumberOfHiddenNeurons() {
        return numberOfHiddenNeurons;
    }

    public void setNumberOfHiddenNeurons(int numberOfHiddenNeurons) {
        this.numberOfHiddenNeurons = numberOfHiddenNeurons;
    }

    public int getNumberOfInputs() {
        return numberOfInputs;
    }

    public void setNumberOfInputs(int numberOfInputs) {
        this.numberOfInputs = numberOfInputs;
    }

    public int getNumberOfOutputs() {
        return numberOfOutputs;
    }

    public void setNumberOfOutputs(int numberOfOutputs) {
        this.numberOfOutputs = numberOfOutputs;
    }

    private double getHi(int i) {
        double opt = hiddenLayerAF.activate(hi[i]);
        return opt;
    }

    private double getKi(int k) {
        return outputLayerAF.activate(ki[k]);
    }

    private double getKiPrime(int k){
        return outputLayerAF.derivative(ki[k]);
    }

    private double getHiPrime(int i){
        return hiddenLayerAF.derivative(hi[i]);
    }

    public double getBias() {
        return bias;
    }

    public double getBiasForK(int k){
        return bias * weightKI[k][0];
    }

    public double getBiasForI(int i){
        return bias * weightIJ[i][0];
    }

    public void setBias(double bias) {
        this.bias = bias;
    }
}
