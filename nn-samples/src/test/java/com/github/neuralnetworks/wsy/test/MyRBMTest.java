package com.github.neuralnetworks.wsy.test;

import java.io.IOException;
import java.io.RandomAccessFile;

import com.amd.aparapi.Kernel.EXECUTION_MODE;
import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.architecture.types.NNFactory;
import com.github.neuralnetworks.architecture.types.RBM;
import com.github.neuralnetworks.calculation.RBMLayerCalculator;
import com.github.neuralnetworks.calculation.neuronfunctions.BernoulliDistribution;
import com.github.neuralnetworks.calculation.neuronfunctions.ConnectionCalculatorFullyConnected;
import com.github.neuralnetworks.input.MultipleNeuronsOutputError;
import com.github.neuralnetworks.test.SimpleInputProvider;
import com.github.neuralnetworks.training.TrainerFactory;
import com.github.neuralnetworks.training.TrainingInputProvider;
import com.github.neuralnetworks.training.random.MersenneTwisterRandomInitializer;
import com.github.neuralnetworks.training.random.NNRandomInitializer;
import com.github.neuralnetworks.training.rbm.AparapiCDTrainer;
import com.github.neuralnetworks.util.Environment;
import libsvm.svm;

public class MyRBMTest {
	MyRBMTest(){
		
	}
	public void testTrainning(String trainFile,int visiableSize,int hiddenSize){
		System.out.println("RBMTest.testTrainning");
		RBM rbm = NNFactory.rbm(visiableSize, hiddenSize, false);
		
//		TrainingInputProvider trainInputProvider = new SimpleInputProvider(new float[][] {{1, 1, 1, 0, 0, 0}, {1, 0, 1, 0, 0, 0}, {0, 1, 1, 1, 0, 0}, {0, 0, 0, 1, 1, 1}, {0, 0, 1, 1, 1, 0}, {0, 0, 0, 1, 0, 1} }, null, 600, 1);
		//TrainingInputProvider testInputProvider =  new SimpleInputProvider(new float[][] {{1, 1, 1, 0, 0, 0}, {1, 0, 1, 0, 0, 0}, {0, 1, 1, 1, 0, 0}, {0, 0, 0, 1, 1, 1}, {0, 0, 1, 1, 1, 0}, {0, 0, 0, 1, 0, 1} }, new float[][] {{1, 0}, {1, 0}, {1, 0}, {0, 1}, {0, 1}, {0, 1} }, 6, 1);
		MultipleNeuronsOutputError error = new MultipleNeuronsOutputError();
		
		//AparapiCDTrainer t = TrainerFactory.cdSigmoidTrainer(rbm, null, null, error, new NNRandomInitializer(new MersenneTwisterRandomInitializer(-0.01f, 0.01f)), 0.02f, 0.5f, 0f, 0f, 1, true);
		MyAparapiCDTrainer trainer= MyTrainerFactory.myCDSigmoidTrainer(rbm, null, null, error, new NNRandomInitializer(new MersenneTwisterRandomInitializer(-0.01f, 0.01f)), 0.02f, 0.5f, 0f, 0f, 1, true);
		MyTrainProcess trainProcess=new MyTrainProcess(trainer,trainFile,visiableSize);
		trainer.setTrainProcess(trainProcess);
		trainer.setLayerCalculator(NNFactory.rbmSigmoidSigmoid(rbm));
		
		Environment.getInstance().setExecutionMode(EXECUTION_MODE.SEQ);
		
		System.out.println("\ntrainning");
		trainer.train();
		System.out.println("trained");
		
/*		Matrix cg1;
		cg1 = rbm.getMainConnections().getConnectionGraph();
		Matrix visible = new Matrix(new float[] { 0.89f, 0.34f, 0.47f, 0.19f }, 1);
		Matrix hidden = new Matrix(2, 1);
		RBMLayerCalculator lc = (RBMLayerCalculator) rbm.getLayerCalculator();
		lc.calculateHiddenLayer(rbm, visible, hidden);*/
	}
	public void trainSVM(){
	}
	public void trainRBM(String trainFile,int visiableSize,int hiddenSize){
		int item=0;
		try{
		    RandomAccessFile inputFile = new RandomAccessFile(trainFile, "r");
			String str;
			while((str=inputFile.readLine())!=null){
				if(item%1000==0)
					System.out.println("Trainning the No."+item+"item in the trainning corpus.....");
			
				String str2;
				int num;
				int pos1,pos2;
				int hash;
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
					num=Integer.parseInt(str2);
					pos1=pos2+1;
					
					pos2=str.indexOf(" " ,pos1);
					if(pos2!=-1){
	    				str2=str.substring(pos1, pos2);
	    				num=Integer.parseInt(str2);
	    				pos1=pos2+1;
					}
					else{
						pos2=str.length();
	    				str2=str.substring(pos1, pos2);
	    				num=Integer.parseInt(str2);
					}
				}
				item++;
			}
		}catch(IOException e){
			e.printStackTrace();
		}
		
	}
	public static void main(String args[]){
		MyRBMTest test=new MyRBMTest();
		test.testTrainning("MiniTrain3000.csv",3001,601);
	}

}
