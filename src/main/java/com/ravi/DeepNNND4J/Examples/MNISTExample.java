package com.ravi.DeepNNND4J.Examples;

import com.ravi.Utils.MNIST;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.NDArrayIndex;

/**
 * Created by ravik on 24/04/2017.
 */
public class MNISTExample
{
    public static void main(String[] args) {
        MNIST mnist = new MNIST(100);
        //System.out.println(mnist.getTestInput());




        System.out.println(mnist.getTrainingInput().get(NDArrayIndex.interval(500,525), NDArrayIndex.all()));

    }
}
