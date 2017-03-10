package com.ravi.BackProp.Utils;

/**
 * Created by 611445924 on 10/03/2017.
 */
public class Logger {
    public static void log(String  msg){
        System.out.println(msg);
    }

    public static void debugLog(String msg){
        if(Constants.debugMode) {
            System.out.println(msg);
        }
    }
}
