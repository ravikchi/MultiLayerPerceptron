package com.ravi.MultiLayerPerceptron.Error;

/**
 * Created by 611445924 on 08/03/2017.
 */
public class LinearEF implements ErrorFunction {

    public double error(double[] desiredOutputs, double[] actOutputs) {
        int n=desiredOutputs.length;
        double error = 0.0;
        for(int t=0; t<n; t++){
            error = error + desiredOutputs[t] - actOutputs[t];
        }
        return error;
    }
}
