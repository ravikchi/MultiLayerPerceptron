package com.ravi.BackProp.Examples;

import com.ravi.BackProp.AF.LinearAF;
import com.ravi.BackProp.AF.SigmoidAF;
import com.ravi.BackProp.Error.RMSE;
import com.ravi.BackProp.Learning.LearningAlgorithm;
import com.ravi.BackProp.Learning.OnlineLearning;
import com.ravi.BackProp.MLP.MLPerceptron;
import com.ravi.BackProp.Training.BackPropagration;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by 611445924 on 10/03/2017.
 */
public class SineLearning {
    public static void main(String[] args){
        double interval= 2*Math.PI/9;
        List<Double> values = new ArrayList<Double>();
        for(double i=0; i<2.5; i=i+0.01){
            values.add(i);
        }

        double[][] inputs = new double[values.size()][1];
        double[][] outputs = new double[values.size()][1];
        int i=0;
        for(Double d : values) {
            inputs[i][0] = d;
            outputs[i][0] = Math.sin(2*Math.PI*d);

            System.out.println("Input "+inputs[i][0]);
            System.out.println("Output "+outputs[i][0]);
            i++;
        }

        MLPerceptron perceptron = new MLPerceptron(new LinearAF(),new SigmoidAF(1.0), 5, 1, 1);

        LearningAlgorithm learn = new OnlineLearning(0.2, new BackPropagration(0.1, 0.01), new RMSE(), perceptron);
        learn.train(inputs, outputs);

        interval= 2*Math.PI/9;
        values = new ArrayList<Double>();
        for(double j=0; j< Math.PI; j=j+interval){
            values.add(j);
        }

        inputs = new double[values.size()][1];
        outputs = new double[values.size()][1];
        i=0;
        for(Double d : values) {
            inputs[i][0] = d;
            outputs[i][0] = Math.sin(2*Math.PI*d);

            //System.out.println(inputs[i][0]);
            //System.out.println(outputs[i][0]);
            i++;
        }

        for(int t=0; t<inputs.length; t++) {
            double[] output = perceptron.getOutput(inputs[t]);
            System.out.println("NN Output :"+output[0]);
            System.out.println("Act Output :"+outputs[t][0]);
        }
    }
}
