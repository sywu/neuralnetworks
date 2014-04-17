package com.github.neuralnetworks.wsy.test;

import com.github.neuralnetworks.architecture.types.NNFactory;
import com.github.neuralnetworks.architecture.types.RBM;
import com.github.neuralnetworks.calculation.OutputError;
import com.github.neuralnetworks.calculation.RBMLayerCalculator;
import com.github.neuralnetworks.calculation.neuronfunctions.BernoulliDistribution;
import com.github.neuralnetworks.calculation.neuronfunctions.ConnectionCalculatorFullyConnected;
import com.github.neuralnetworks.training.TrainerFactory;
import com.github.neuralnetworks.training.TrainingInputProvider;
import com.github.neuralnetworks.training.random.NNRandomInitializer;
import com.github.neuralnetworks.training.rbm.AparapiCDTrainer;

public class MyTrainerFactory extends TrainerFactory{
	public static MyAparapiCDTrainer myCDSigmoidTrainer(RBM rbm, TrainingInputProvider trainingSet, TrainingInputProvider testingSet, OutputError error, NNRandomInitializer rand, float learningRate, float momentum, float l1weightDecay, float l2weightDecay, int gibbsSampling, boolean isPersistentCD) {
		rbm.setLayerCalculator(NNFactory.rbmSigmoidSigmoid(rbm));

		RBMLayerCalculator lc = NNFactory.rbmSigmoidSigmoid(rbm);
		ConnectionCalculatorFullyConnected cc = (ConnectionCalculatorFullyConnected) lc.getConnectionCalculator(rbm.getInputLayer());
		cc.addPreTransferFunction(new BernoulliDistribution());

		return new MyAparapiCDTrainer(rbmProperties(rbm, lc, trainingSet, testingSet, error, rand, learningRate, momentum, l1weightDecay, l2weightDecay, gibbsSampling, isPersistentCD));
	}

}
