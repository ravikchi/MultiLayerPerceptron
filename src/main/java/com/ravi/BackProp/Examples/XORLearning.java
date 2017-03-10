package com.ravi.BackProp.Examples;

import com.ravi.BackProp.AF.LinearAF;
import com.ravi.BackProp.AF.SigmoidAF;
import com.ravi.BackProp.Error.LinearEF;
import com.ravi.BackProp.Error.RMSE;
import com.ravi.BackProp.Learning.LearningAlgorithm;
import com.ravi.BackProp.Learning.OnlineLearning;
import com.ravi.BackProp.MLP.MLPerceptron;
import com.ravi.BackProp.Training.BackPropagration;

import java.text.DecimalFormat;

/**
 * Created by 611445924 on 08/03/2017.
 */
public class XORLearning {
    public static void main(String[] args){
        double[][] inputs = new double[4][2];
        inputs[0][0] = 1.0;
        inputs[0][1] = 1.0;

        inputs[1][0] = 0.0;
        inputs[1][1] = 1.0;

        inputs[2][0] = 1.0;
        inputs[2][1] = 0.0;

        inputs[3][0] = 0.0;
        inputs[3][1] = 0.0;

        double[][] outputs = new double[4][1];
        outputs[0][0] = 0.0;
       outputs[1][0] = 1.0;
        outputs[2][0] = 1.0;
        outputs[3][0] = 0.0;

        MLPerceptron perceptron = new MLPerceptron(new LinearAF(),new SigmoidAF(1.0), 2, 2, 1);

        LearningAlgorithm learn = new OnlineLearning(0.0, new BackPropagration(0.1, 0.01), new RMSE(), perceptron);
        learn.train(inputs, outputs);

        DecimalFormat df2 = new DecimalFormat(".#####");

        for(int t=0; t<inputs.length; t++){
            System.out.println(df2.format(perceptron.getOutput(inputs[t])[0]));
        }
    }
}
