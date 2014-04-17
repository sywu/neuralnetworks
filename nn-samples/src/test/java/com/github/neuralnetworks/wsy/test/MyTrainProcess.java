package com.github.neuralnetworks.wsy.test;

import java.io.IOException;
import java.io.RandomAccessFile;

import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.architecture.types.RBM;
import com.github.neuralnetworks.training.TrainingInputData;

public class MyTrainProcess {
	private MyOneStepTrainer<RBM> trainer=null;
	private String trainFile=null;
	private static int miniBatchSize = 1;
	private int visiableSize=0;
	
	MyTrainProcess(MyOneStepTrainer<RBM> trainer,String trainFile,int visiableSize){
		this.trainer=trainer;
		this.trainFile=trainFile;
		this.visiableSize=visiableSize;
	}
	public void train(){
		System.out.println("MyTrainProcess.train");  
		int batch = 0;
	    
		RandomAccessFile inputFile;
		int item=0;
		try{
		    inputFile = new RandomAccessFile(trainFile, "r");
			String str;
			long st=System.currentTimeMillis();
			while((str=inputFile.readLine())!=null && !trainer.isStopTraining()){
				if(item%10==0)
					System.out.println("Trainning the No."+item+"item in the trainning corpus.....("+(System.currentTimeMillis()-st)+"s)");

				MySimpleTrainingInputData input = new MySimpleTrainingInputData(null, null);
			    input.setInput(new Matrix(visiableSize, miniBatchSize));
			    
				String str2;
				int num1,num2;
				int pos1,pos2;
				pos1=0;
				//==============================读入类别====================================
				while((pos2=str.indexOf(", " ,pos1))!=-1){
	/*				str2=str.substring(pos1, pos2);
					num=Integer.parseInt(str2);*/
					pos1=pos2+2;
				}
		
				pos2=str.indexOf(" " ,pos1);
				if(pos2!=-1){
	/*				str2=str.substring(pos1, pos2);
					num=Integer.parseInt(str2);*/
					pos1=pos2+1;
				}
				//==============================读入类别====================================
				while((pos2=str.indexOf(":" ,pos1))!=-1){
					str2=str.substring(pos1, pos2);
					num1=Integer.parseInt(str2);
					pos1=pos2+1;
					
					pos2=str.indexOf(" " ,pos1);
					if(pos2!=-1){
	    				str2=str.substring(pos1, pos2);
	    				num2=Integer.parseInt(str2);

	    				input.getInput().set(num1, 1, num2);
	    				pos1=pos2+1;
					}
					else{
						pos2=str.length();
	    				str2=str.substring(pos1, pos2);
	    				num2=Integer.parseInt(str2);
	    				input.getInput().set(num1, 1, num2);
					}
				}
				trainer.trainIterativeProcess(input, batch);
				batch++;
				item++;
			}
		}catch(IOException e){
			e.printStackTrace();
		}
		
	}
}

class MySimpleTrainingInputData implements TrainingInputData {

	private Matrix input;
	private Matrix target;

	public MySimpleTrainingInputData(Matrix input, Matrix target) {
	    super();
	    this.input = input;
	    this.target = target;
	}

	@Override
	public Matrix getInput() {
	    return input;
	}

	public void setInput(Matrix input) {
	    this.input = input;
	}

	@Override
	public Matrix getTarget() {
	    return target;
	}

	public void setTarget(Matrix target) {
	    this.target = target;
   }
}

