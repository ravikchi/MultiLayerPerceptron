package com.ravi.BackProp.Examples;

import com.ravi.BackProp.AF.LinearAF;
import com.ravi.BackProp.AF.SigmoidAF;
import com.ravi.BackProp.AF.StepFunction;
import com.ravi.BackProp.MLP.MLPerceptron;

/**
 * Created by 611445924 on 08/03/2017.
 */
public class XORPerceptron {
    public static void main(String[] args){
        MLPerceptron perceptron = new MLPerceptron(new LinearAF(),new SigmoidAF(1.0), 2, 2, 1);

        /*double[][] weightsIJ = new double[2][3];
        weightsIJ[0][0] = -0.5;
        weightsIJ[0][1] = 1.0;
        weightsIJ[0][2] = -1.0;

        weightsIJ[1][0] = -0.5;
        weightsIJ[1][1] = -1.0;
        weightsIJ[1][2] = 1.0;


        perceptron.setWeightIJ(weightsIJ);

        double[][] weightsKI = new double[1][3];
        weightsKI[0][0] = -0.5;
        weightsKI[0][1] = 1.0;
        weightsKI[0][2] = 1.0;

        perceptron.setWeightKI(weightsKI);

        double[][] inputs = new double[4][2];
        inputs[0][0] = 1.0;
        inputs[0][1] = 1.0;

        inputs[1][0] = 0.0;
        inputs[1][1] = 1.0;

        inputs[2][0] = 1.0;
        inputs[2][1] = 0.0;

        inputs[3][0] = 0.0;
        inputs[3][1] = 0.0;*/

        double[][] weightsIJ = new double[2][3];
        weightsIJ[0][0] = 0.35;
        weightsIJ[0][1] = 0.4;
        weightsIJ[0][2] = 0.7;

        weightsIJ[1][0] = 0.42;
        weightsIJ[1][1] = 0.3;
        weightsIJ[1][2] = 0.9;


        perceptron.setWeightIJ(weightsIJ);

        double[][] weightsKI = new double[1][3];
        weightsKI[0][0] = 0.7;
        weightsKI[0][1] = 0.1;
        weightsKI[0][2] = 0.35;

        perceptron.setWeightKI(weightsKI);

        double[][] inputs = new double[4][2];
        inputs[0][0] = 1.0;
        inputs[0][1] = 1.0;

        inputs[1][0] = 0.0;
        inputs[1][1] = 1.0;

        inputs[2][0] = 1.0;
        inputs[2][1] = 0.0;

        inputs[3][0] = 0.0;
        inputs[3][1] = 0.0;

        double[][] outputs = new double[4][2];
        outputs[0][0] = 0.0;
        outputs[1][0] = 1.0;
        outputs[2][0] = 1.0;
        outputs[3][0] = 0.0;

        for(int t=0; t<inputs.length; t++){
            double[] actOutputs = perceptron.getOutput(inputs[t]);
            double ekt = outputs[t][0]-actOutputs[0];
            System.out.println(ekt);
            System.out.println(actOutputs[0]);
        }

        SigmoidAF af = new SigmoidAF(1.0);
        System.out.println("Activation Function"+af.activate(0.78));
    }
}
