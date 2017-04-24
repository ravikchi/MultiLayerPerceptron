package com.ravi.DeepNNND4J.Examples;

import com.ravi.DeepNNND4J.MNIST;

/**
 * Created by ravik on 24/04/2017.
 */
public class MNISTExample
{
    public static void main(String[] args) {
        MNIST mnist = new MNIST(10000);
        System.out.println(mnist.getTestInput());
    }
}
