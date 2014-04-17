package com.github.neuralnetworks.wsy.test;

import static org.junit.Assert.assertEquals;

import com.amd.aparapi.Kernel;
import com.amd.aparapi.Kernel.EXECUTION_MODE;
import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.architecture.NeuralNetworkImpl;
import com.github.neuralnetworks.architecture.types.NNFactory;
import com.github.neuralnetworks.architecture.types.RBM;
import com.github.neuralnetworks.calculation.RBMLayerCalculator;
import com.github.neuralnetworks.input.MultipleNeuronsOutputError;
import com.github.neuralnetworks.input.ScalingInputFunction;
import com.github.neuralnetworks.samples.mnist.MnistInputProvider;
import com.github.neuralnetworks.samples.mnist.MnistTargetMultiNeuronOutputConverter;
import com.github.neuralnetworks.test.SimpleInputProvider;
import com.github.neuralnetworks.training.TrainerFactory;
import com.github.neuralnetworks.training.TrainingInputProvider;
import com.github.neuralnetworks.training.backpropagation.BackPropagationTrainer;
import com.github.neuralnetworks.training.events.LogTrainingListener;
import com.github.neuralnetworks.training.random.MersenneTwisterRandomInitializer;
import com.github.neuralnetworks.training.random.NNRandomInitializer;
import com.github.neuralnetworks.training.rbm.AparapiCDTrainer;
import com.github.neuralnetworks.util.Environment;

public class MyTest {
	private void testMnistTest(){
		NeuralNetworkImpl mlp = NNFactory.mlpSigmoid(new int[] {  784, 10 }, true);
		MnistInputProvider trainInputProvider = new MnistInputProvider("train-images.idx3-ubyte", "train-labels.idx1-ubyte", 1, 1, new MnistTargetMultiNeuronOutputConverter());
		trainInputProvider.addInputModifier(new ScalingInputFunction(255));
		//NeuralNetworkImpl mlp = NNFactory.mlpSigmoid(new int[] { 1617900, 2 }, true);
		//MyMnistInputProvider trainInputProvider = new MyMnistInputProvider("DataSet/train-remapped/vectors", "DataSet/train/cats",2365436, 1, 1, new MnistTargetMultiNeuronOutputConverter());
		//trainInputProvider.addInputModifier(new ScalingInputFunction(1700));
		MnistInputProvider testInputProvider = new MnistInputProvider("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte", 1000, 1, new MnistTargetMultiNeuronOutputConverter());
		testInputProvider.addInputModifier(new ScalingInputFunction(255));

		BackPropagationTrainer<?> bpt = TrainerFactory.backPropagation(mlp, trainInputProvider, testInputProvider, new MultipleNeuronsOutputError(), new NNRandomInitializer(new MersenneTwisterRandomInitializer(-0.01f, 0.01f)), 0.02f, 0.5f, 0f, 0f);

		bpt.addEventListener(new LogTrainingListener(Thread.currentThread().getStackTrace()[1].getMethodName(), false, true));

		Environment.getInstance().setExecutionMode(EXECUTION_MODE.GPU);

		bpt.train();
		//bpt.test();
	}

    public void testPCD() {//PersistentContrastiveDivergence
		System.out.println("testPCD");
		RBM rbm = NNFactory.rbm(6, 2, false);
		
		TrainingInputProvider trainInputProvider = new SimpleInputProvider(new float[][] {{1, 1, 1, 0, 0, 0}, {1, 0, 1, 0, 0, 0}, {0, 1, 1, 1, 0, 0}, {0, 0, 0, 1, 1, 1}, {0, 0, 1, 1, 1, 0}, {0, 0, 0, 1, 0, 1} }, null, 600, 1);
		TrainingInputProvider testInputProvider =  new SimpleInputProvider(new float[][] {{1, 1, 1, 0, 0, 0}, {1, 0, 1, 0, 0, 0}, {0, 1, 1, 1, 0, 0}, {0, 0, 0, 1, 1, 1}, {0, 0, 1, 1, 1, 0}, {0, 0, 0, 1, 0, 1} }, new float[][] {{1, 0}, {1, 0}, {1, 0}, {0, 1}, {0, 1}, {0, 1} }, 6, 1);
		MultipleNeuronsOutputError error = new MultipleNeuronsOutputError();
		
		AparapiCDTrainer t = TrainerFactory.cdSigmoidTrainer(rbm, trainInputProvider, testInputProvider, error, new NNRandomInitializer(new MersenneTwisterRandomInitializer(-0.01f, 0.01f)), 0.02f, 0.5f, 0f, 0f, 1, true);
		t.setLayerCalculator(NNFactory.rbmSigmoidSigmoid(rbm));
		
		Environment.getInstance().setExecutionMode(EXECUTION_MODE.GPU);
		Matrix cg1;
		int col,row;
		cg1 = rbm.getMainConnections().getConnectionGraph();
		col=cg1.getColumns();
		row=cg1.getRows();
		for(int i=0;i<row;i++){
			for(int j=0;j<col;j++)
				System.out.print(cg1.get(i, j)+" ");
			System.out.println();
		}
		
		System.out.println("\ntrainning");
		t.train();
		System.out.println("trained");
		cg1 = rbm.getMainConnections().getConnectionGraph();
		col=cg1.getColumns();
		row=cg1.getRows();
		for(int i=0;i<row;i++){
			for(int j=0;j<col;j++)
				System.out.print(cg1.get(i, j)+" ");
			System.out.println();
		}
		
		Matrix visible = new Matrix(new float[] { 0.89f, 0.34f, 0.47f, 0.19f }, 1);
		Matrix hidden = new Matrix(2, 1);
		RBMLayerCalculator lc = (RBMLayerCalculator) rbm.getLayerCalculator();
		lc.calculateHiddenLayer(rbm, visible, hidden);
		
		for(int i=0;i<row;i++)
			System.out.print(hidden.get(i, 0)+" ");
		System.out.println();
		
		//cg1 = rbm.getMainConnections().getConnectionGraph();
		//System.out.println(cg1.toString());
		
		//assertEquals(0, t.getOutputError().getTotalNetworkError(), 0);
    }
	private void testGPU(){
		System.out.println("testGPU");
		int[] size=new int[]{1000,10000,100000,1000000,10000000,100000000};
		long timeSt,timeEd;
		MatrixCalcKernel kernel=null;
		for(int i=0;i<size.length;i++){
			System.out.println("Size "+size[i]);
			kernel=new MatrixCalcKernel(size[i],size[i],EXECUTION_MODE.GPU);
			timeSt=System.currentTimeMillis();
			kernel.calc();
			timeEd=System.currentTimeMillis();
			System.out.println("GPU:  "+(timeEd-timeSt)+"\n");
		}
		for(int i=0;i<size.length;i++){
			System.out.println("Size "+size[i]);
			kernel=new MatrixCalcKernel(size[i],size[i],EXECUTION_MODE.CPU);
			timeSt=System.currentTimeMillis();
			kernel.calc();
			timeEd=System.currentTimeMillis();
			System.out.println("CPU:  "+(timeEd-timeSt)+"\n");
		}
	}
	public static void main(String args[]){
		MyTest test=new MyTest();
		//test.testMnistTest();
		//test.testGPU();
		test.testPCD();
	}

}
class MatrixCalcKernel extends Kernel {
	private EXECUTION_MODE mode;

	private float[] x,y,z;

	private float[][] a,ax;
	
	private int n,m;

	public MatrixCalcKernel(int n,int m,EXECUTION_MODE mode) {
	    super();
	    this.mode=mode;
	    this.n=n;
	    this.m=m;
	    this.x = new float[n];
	    this.y=new float[n];
	    this.z=new float[n*n];
	    for (int i = 0; i < n; i++){
	    	x[i] = i+1;
	    	y[i] =i+1;
	    }
	}

	public void calc() {
	    setExecutionMode(mode);
	    setExplicit(true);
	    execute(n*n);
	}

	@Override
	public void run() {
	    int i = getGlobalId();
	    float temp=x[i/n]*y[i%n];
	    /*
	    if((j+1)%(m/20<1?1:m/20)==0)
	    	System.out.println("Calculation complete ....."+(j+1)*100/m+"%");
	    	*/
	}
}
