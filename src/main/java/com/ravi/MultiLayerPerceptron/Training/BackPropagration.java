package com.ravi.MultiLayerPerceptron.Training;

import com.ravi.MultiLayerPerceptron.MLP.MLPerceptron;

/**
 * Created by 611445924 on 08/03/2017.
 */
public class BackPropagration implements TrainingAlgorithm {
    private MLPerceptron perceptron;
    private double[][] oldDWKI;
    private double[][] oldDWIJ;
    private double learningRate;
    private double alpha;


    public BackPropagration(double learningRate, double alpha) {
        this.learningRate = learningRate;
        this.alpha = alpha;
    }

    public void runEpoch(double[] inputs, double ekt, MLPerceptron perceptron, int t) {
        this.perceptron = perceptron;

        if(t==0) {
            oldDWKI = new double[perceptron.getNumberOfOutputs()][perceptron.getNumberOfHiddenNeurons()+1];
            oldDWIJ = new double[perceptron.getNumberOfHiddenNeurons()][perceptron.getNumberOfInputs()+1];
        }

        double[][] weightsKI = perceptron.getWeightKI();

        double[] deltaK = new double[perceptron.getNumberOfOutputs()];

        for(int k=0; k<perceptron.getNumberOfOutputs(); k++){
            deltaK[k] = deltaK(inputs, k, ekt);
        }

        for(int k=0; k<perceptron.getNumberOfOutputs(); k++){
            for(int i=0; i<perceptron.getNumberOfHiddenNeurons()+1; i++) {
                weightsKI[k][i] = weightsKI[k][i] + deltaWKI(inputs, k, i, ekt, deltaK[k]);
                oldDWKI[k][i] = deltaWKI(inputs, k, i, ekt, deltaK[k]);
            }
        }

        perceptron.setWeightKI(weightsKI);

        double[][] weightsIJ = perceptron.getWeightIJ();

        double[] deltaI = new double[perceptron.getNumberOfHiddenNeurons()];

        for(int i=0; i<perceptron.getNumberOfHiddenNeurons(); i++){
            deltaI[i] = deltaI(inputs, i, ekt);
        }

        for(int i=0; i<perceptron.getNumberOfHiddenNeurons(); i++){
            for(int j=0; j<perceptron.getNumberOfInputs()+1; j++){
                weightsIJ[i][j] = weightsIJ[i][j] + deltaWHJ(inputs, i, j, ekt, deltaI[i]);
                oldDWIJ[i][j] = deltaWHJ(inputs, i, j, ekt, deltaI[i]);
            }
        }

        perceptron.setWeightIJ(weightsIJ);
    }

    private double deltaWKI(double[] inputs, int k, int i, double ekt, double deltaK){
        double hi = 0.0;
        if(i==0){
            hi = perceptron.getBiasForK(k);
        }else{
            hi = perceptron.getHi(inputs, i-1);
        }

        double learning = (learningRate * deltaK * hi);
        double momemtum = (alpha * oldDWKI[k][i]);
        return  learning + momemtum;
    }

    private double deltaWHJ(double[] inputs, int i, int j, double ekt, double deltaI){
        double ij = 0.0;
        if(j==0){
            ij = perceptron.getBias();
        }else{
            ij = inputs[j-1];
        }
        double learning = learningRate * deltaI * ij;
        double momemtum = alpha * oldDWIJ[i][j];
        return learning + momemtum;
    }

    private double deltaK(double[] inputs, int k, double ekt){
        return perceptron.getKiPrime(inputs, k) * ekt;
    }

    private double deltaI(double[] inputs, int i, double ekt){
        double hPrime = perceptron.getHiPrime(inputs, i);
        double sumOfDeltaW = calculateDeltaKWI(inputs, ekt, i+1);
        return hPrime * sumOfDeltaW;
    }

    private double calculateDeltaKWI(double[] inputs, double ekt, int i){
        double deltaW = 0.0;
        for(int k=0; k<perceptron.getNumberOfOutputs();k++){
            double dki= deltaK(inputs, k, ekt);
            double weight = perceptron.getWeightKI()[k][i];
            deltaW = deltaW + dki * weight;
        }

        return deltaW;
    }
}
