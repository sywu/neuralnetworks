package com.github.neuralnetworks.wsy.test;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.Random;


public class MyGenerateMiniData {
    private Random random;
	MyGenerateMiniData(){
		
	}
	public void generateMiniData(int itemSize,String fileName,int featureSize,int typeSize,double p){
		File outFile;
		int[] feature;
	    random = new Random();
	    if(fileName==null)
	    	outFile=new File("miniTrain"+itemSize+".csv");
	    else
	    	outFile=new File(fileName);
		try {
			outFile.createNewFile();
			FileOutputStream fis=new FileOutputStream(outFile);
			OutputStreamWriter osw=new OutputStreamWriter(fis);
			for(int i=0;i<itemSize;i++){
				feature=new int[featureSize];
				for(int j=0;j<featureSize;j++)
					if(random.nextDouble()-p<0.0001)
						feature[j]=1;
				
				String output="";
				for(int l=1;l<=typeSize;l++){
					int st,len,ed;
					len=featureSize/typeSize;
					st=len*(l-1);
					ed=st+len;
					for(int j=st;j<ed;j++)
						if(feature[j]==1){
							if(output.length()==0)
								output+=Integer.toString(l);
							else
								output+=", "+Integer.toString(l);
							break;
						}
				}
				if(output.length()==0)
					output+=Integer.toString(typeSize+1);
				for(int j=0;j<featureSize;j++)
					if(feature[j]==1)
						output+=" "+Integer.toString(j)+":"+Integer.toString(1);
				osw.write(output+"\n");
			}
			osw.close();	
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	public static void main(String args[]){
		MyGenerateMiniData generator=new MyGenerateMiniData();
		generator.generateMiniData(10000, "MiniTrain3000.csv", 3000, 600, 0.003);
		generator.generateMiniData(1000, "MiniTest3000.csv", 3000, 600, 0.003);
	}
}
