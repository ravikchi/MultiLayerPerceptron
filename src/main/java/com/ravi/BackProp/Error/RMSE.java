package com.ravi.BackProp.Error;

/**
 * Created by 611445924 on 08/03/2017.
 */
public class RMSE implements ErrorFunction {

    public double error(double[] desiredOutputs, double[] actOutputs) {
        int n=desiredOutputs.length;
        double error = 0.0;
        for(int t=0; t<n; t++){
            double tError = desiredOutputs[t] - actOutputs[t];
            error = error + tError * tError;
        }
        error = error/n;
        return Math.sqrt(error);
    }
}
