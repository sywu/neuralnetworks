package com.github.neuralnetworks.wsy.test;

import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.architecture.types.RBM;
import com.github.neuralnetworks.training.rbm.CDBiasUpdatesKernel;
import com.github.neuralnetworks.training.rbm.CDWeightUpdatesKernel;
import com.github.neuralnetworks.util.Constants;
import com.github.neuralnetworks.util.Environment;
import com.github.neuralnetworks.util.Properties;

public class MyAparapiCDTrainer extends MyCDTrainerBase {

    /**
     * weights update kernel for the connections between the visible and the hidden layer
     */
    private MyCDWeightUpdatesKernel weightUpdatesKernel;
    //private CDWeightUpdatesKernel weightUpdatesKernel;

    /**
     * weights update kernel for visible bias connections
     */
    private CDBiasUpdatesKernel visibleBiasUpdatesKernel;

    /**
     * weights update kernel for the hidden bias connections
     */
    private CDBiasUpdatesKernel hiddenBiasUpdatesKernel;

    public MyAparapiCDTrainer(Properties properties) {
    	super(properties);
    }


    /* (non-Javadoc)
     * @see com.github.neuralnetworks.training.rbm.CDTrainerBase#updateWeights(com.github.neuralnetworks.architecture.Matrix, com.github.neuralnetworks.architecture.Matrix, com.github.neuralnetworks.architecture.Matrix, com.github.neuralnetworks.architecture.Matrix)
     * before each update the kernel update parameters are refreshed
     */
    @Override
    protected void updateWeights(Matrix posPhaseVisible, Matrix posPhaseHidden, Matrix negPhaseVisible, Matrix negPhaseHidden) {
	RBM rbm = getNeuralNetwork();

	int mbs = posPhaseHidden.getColumns();

	if (weightUpdatesKernel == null || weightUpdatesKernel.getMiniBatchSize() != mbs) {
	    weightUpdatesKernel = new MyCDWeightUpdatesKernel(posPhaseVisible.getElements(), posPhaseHidden.getElements(), negPhaseVisible.getElements(), negPhaseHidden.getElements(), rbm.getMainConnections().getConnectionGraph().getElements(), rbm.getMainConnections().getConnectionGraph().getColumns(), getLearningRate(), getMomentum(), getl1weightDecay(), getl2weightDecay(), mbs);
	    
	}
	Environment.getInstance().getExecutionStrategy().execute(weightUpdatesKernel, rbm.getMainConnections().getConnectionGraph().getRows()*rbm.getMainConnections().getConnectionGraph().getColumns());


/*	if (weightUpdatesKernel == null || weightUpdatesKernel.getMiniBatchSize() != mbs) {
	    weightUpdatesKernel = new CDWeightUpdatesKernel(posPhaseVisible.getElements(), posPhaseHidden.getElements(), negPhaseVisible.getElements(), negPhaseHidden.getElements(), rbm.getMainConnections().getConnectionGraph().getElements(), rbm.getMainConnections().getConnectionGraph().getColumns(), getLearningRate(), getMomentum(), getl1weightDecay(), getl2weightDecay(), mbs);
	    
	}
	Environment.getInstance().getExecutionStrategy().execute(weightUpdatesKernel, rbm.getMainConnections().getConnectionGraph().getRows());*/

	// update visible bias
	if (rbm.getVisibleBiasConnections() != null) {
	    if (visibleBiasUpdatesKernel == null || visibleBiasUpdatesKernel.getMiniBatchSize() != mbs) {
		visibleBiasUpdatesKernel = new CDBiasUpdatesKernel(rbm.getVisibleBiasConnections().getConnectionGraph().getElements(), posPhaseVisible.getElements(), negPhaseVisible.getElements(), getLearningRate(), getMomentum(), mbs);
	    }

	    Environment.getInstance().getExecutionStrategy().execute(visibleBiasUpdatesKernel, rbm.getVisibleBiasConnections().getConnectionGraph().getElements().length);
	}

	// update hidden bias
	if (rbm.getHiddenBiasConnections() != null) {
	    if (hiddenBiasUpdatesKernel == null || hiddenBiasUpdatesKernel.getMiniBatchSize() != mbs) {
		hiddenBiasUpdatesKernel = new CDBiasUpdatesKernel(rbm.getHiddenBiasConnections().getConnectionGraph().getElements(), posPhaseHidden.getElements(), negPhaseHidden.getElements(), getLearningRate(), getMomentum(), mbs);
	    }

	    Environment.getInstance().getExecutionStrategy().execute(hiddenBiasUpdatesKernel, rbm.getHiddenBiasConnections().getConnectionGraph().getElements().length);
	}
    }

    protected float getLearningRate() {
	return properties.getParameter(Constants.LEARNING_RATE);
    }
    
    protected float getMomentum() {
	return (float) (properties.getParameter(Constants.MOMENTUM) != null ? properties.getParameter(Constants.MOMENTUM) : 0f);
    }

    protected float getl1weightDecay() {
	return (float) (properties.getParameter(Constants.L1_WEIGHT_DECAY) != null ? properties.getParameter(Constants.L1_WEIGHT_DECAY) : 0f);
    }
    
    protected float getl2weightDecay() {
	return (float) (properties.getParameter(Constants.L2_WEIGHT_DECAY) != null ? properties.getParameter(Constants.L2_WEIGHT_DECAY) : 0f);
    }
}

