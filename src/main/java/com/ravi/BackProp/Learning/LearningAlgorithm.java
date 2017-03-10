package com.ravi.BackProp.Learning;

import com.ravi.BackProp.MLP.MLPerceptron;

/**
 * Created by 611445924 on 08/03/2017.
 */
public interface LearningAlgorithm {
    public void train(double[][] inputs, double[][] outputs);
}
