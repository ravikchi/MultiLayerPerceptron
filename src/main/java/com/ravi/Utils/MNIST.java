package com.ravi.Utils;

import au.com.bytecode.opencsv.CSVReader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

/**
 * Created by ravik on 24/04/2017.
 */
public class MNIST {
    private INDArray trainingOutput;
    private INDArray trainingInput;

    INDArray testOutput;
    INDArray testInput;

    public void fetchData(boolean training, int numOfExamples, String path){
        try {
            CSVReader reader = new CSVReader(new FileReader(path));
            String[] nextLine = null;

            INDArray output = Nd4j.zeros(1, numOfExamples);
            INDArray input = Nd4j.zeros(784, numOfExamples);

            String[] header = reader.readNext();

            int count=0;
            while((nextLine = reader.readNext()) != null){
                output.put(0, count, Double.parseDouble(nextLine[0]));
                for(int i=1; i<nextLine.length; i++){
                    input.put(i-1, count, Double.parseDouble(nextLine[i]));
                }
                count++;
                if(count>=numOfExamples){
                    break;
                }
            }

            if(training){
                this.setTrainingInput(input);
                this.setTrainingOutput(output);
            }else{
                this.setTestInput(input);
                this.setTestOutput(output);
            }

        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public MNIST(int numOfExamples) {
        fetchData(true, numOfExamples, "C:\\Users\\rc16956\\Downloads\\train.csv");
        fetchData(false, numOfExamples/10, "C:\\Users\\rc16956\\Downloads\\test.csv");
    }

    public INDArray getTestOutput() {
        return testOutput;
    }

    public void setTestOutput(INDArray testOutput) {
        this.testOutput = testOutput;
    }

    public INDArray getTestInput() {
        return testInput;
    }

    public void setTestInput(INDArray testInput) {
        this.testInput = testInput;
    }

    public INDArray getTrainingOutput() {
        return trainingOutput;
    }

    public void setTrainingOutput(INDArray trainingOutput) {
        this.trainingOutput = trainingOutput;
    }

    public INDArray getTrainingInput() {
        return trainingInput;
    }

    public void setTrainingInput(INDArray trainingInput) {
        this.trainingInput = trainingInput;
    }
}
