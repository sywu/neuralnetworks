package com.github.neuralnetworks.wsy.test;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.RandomAccessFile;


public class SamplifyInput {
	final static int BigPrimeNumber=9999943;
	private static int typeHash[]=new int[BigPrimeNumber];
	private static int typeValue[]=new int[BigPrimeNumber];
	private static int featureHash[]=new int[BigPrimeNumber];
	private static int featureValue[]=new int[BigPrimeNumber];
	static int typeNum=0;
	static int featureNum=0;
	private static int hashType(int type){
		int v=type%BigPrimeNumber;
		while(typeHash[v]!=0 && typeHash[v]!=type)
			v=(v+1)%BigPrimeNumber;
		if(typeHash[v]==0){
			typeHash[v]=type;
			typeNum++;
			typeValue[v]=typeNum;
		}
		return typeValue[v];	
	}
	private static int hashFeature(int feature){
		int v=feature%BigPrimeNumber;
		while(featureHash[v]!=0 && featureHash[v]!=feature)
			v=(v+1)%BigPrimeNumber;
		if(featureHash[v]==0){
			featureHash[v]=feature;
			featureNum++;
			featureValue[v]=featureNum;
		}
		return featureValue[v];	
	}
	public static void main(String args[]){
    	int item=0;
    	try{
    	    RandomAccessFile trainFile = new RandomAccessFile("MiniTrain.csv", "r");
    		File outFile=new File("miniTrain2.csv");
    		outFile.createNewFile();
    		FileOutputStream fis=new FileOutputStream(outFile);
    		OutputStreamWriter osw=new OutputStreamWriter(fis);
    		String str;
    		while((str=trainFile.readLine())!=null){
    			if(item>65536)
    				break;
    		
    			String str2;
    			int num;
    			int pos1,pos2;
    			int hash;
    			pos1=0;
    			while((pos2=str.indexOf(", " ,pos1))!=-1){
    				str2=str.substring(pos1, pos2);
    				num=Integer.parseInt(str2);
    				hash=hashType(num);
    				osw.write(hash+", ");
    				pos1=pos2+2;
    			}
    	
    			pos2=str.indexOf(" " ,pos1);
    			if(pos2!=-1){
    				str2=str.substring(pos1, pos2);
    				num=Integer.parseInt(str2);
    				hash=hashType(num);
    				osw.write(hash+" ");
    				pos1=pos2+1;
    			}
    			
    			while((pos2=str.indexOf(":" ,pos1))!=-1){
    				str2=str.substring(pos1, pos2);
    				num=Integer.parseInt(str2);
    				hash=hashFeature(num);
    				osw.write(hash+":");
    				pos1=pos2+1;
    				pos2=str.indexOf(" " ,pos1);
    				if(pos2!=-1){
        				str2=str.substring(pos1, pos2);
        				num=Integer.parseInt(str2);
        				osw.write(num+" ");
        				pos1=pos2+1;
    				}
    				else{
    					pos2=str.length();
        				str2=str.substring(pos1, pos2);
        				num=Integer.parseInt(str2);
        				osw.write(num+"\n");
    				}
    			}
    			item++;
    		}
    		osw.close();
    		System.out.println(typeNum+"\n"+featureNum);
    	}catch(IOException e){
    		e.printStackTrace();
    	}
    	
	}
}
