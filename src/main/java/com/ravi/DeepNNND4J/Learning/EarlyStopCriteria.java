package com.ravi.DeepNNND4J.Learning;

/**
 * Created by 611445924 on 18/04/2017.
 */
public class EarlyStopCriteria {
    double validationSize;
    int maxEpoch;
    double minError;
    double maxValidationCount;

    public EarlyStopCriteria(double validationSize, int maxEpoch, double minError, double maxValidationCount) {
        this.validationSize = validationSize;
        this.maxEpoch = maxEpoch;
        this.minError = minError;
        this.maxValidationCount = maxValidationCount;
    }

    public double getValidationSize() {
        return validationSize;
    }

    public void setValidationSize(double validationSize) {
        this.validationSize = validationSize;
    }

    public int getMaxEpoch() {
        return maxEpoch;
    }

    public void setMaxEpoch(int maxEpoch) {
        this.maxEpoch = maxEpoch;
    }

    public double getMinError() {
        return minError;
    }

    public void setMinError(double minError) {
        this.minError = minError;
    }

    public double getMaxValidationCount() {
        return maxValidationCount;
    }

    public void setMaxValidationCount(double maxValidationCount) {
        this.maxValidationCount = maxValidationCount;
    }
}
