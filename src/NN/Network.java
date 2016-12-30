package NN;

import java.util.Arrays;

public class Network {
	private int num_hidden_layer;
	private int num_hidden_unit;
	private int num_input;
	private int num_output;
	private float stepsize;
	private float stepsize_base;
	private int count_back_prop;
	private int count_unit;
	private Matrix[] weights;
	private Matrix[] interpretations;
	private Matrix[] activations;
	
	public Network(int num_hidden_layer, int num_hidden_unit, 
			int num_input, int num_output, float stepsize) {
		this.num_hidden_layer = num_hidden_layer;
		this.num_hidden_unit = num_hidden_unit;
		this.num_input = num_input;
		this.num_output = num_output;
		this.weights = new Matrix[num_hidden_layer + 1];
		this.stepsize_base = stepsize;
		this.stepsize = this.stepsize_base;
		this.count_back_prop = 0;
		
		// last row is for bias
		this.weights[0] = new Matrix(num_hidden_unit, num_input + 1, true);
		for(int i = 1; i < num_hidden_layer; ++i) {
			this.weights[i] = new Matrix(num_hidden_unit, num_hidden_unit + 1, true);
		}
		this.weights[num_hidden_layer] = new Matrix(num_output, num_hidden_unit + 1, true);
		this.count_unit = 0;
		for(Matrix w : this.weights) {
			this.count_unit += w.getNumCol() * w.getNumRow();
		}
	}
	
	/**
	 * forward interpretation, from feature to score for each class(which is not returned)
	 * @param feature
	 */
	public void forward(float[] feature) {
		this.interpretations = new Matrix[this.num_hidden_layer + 1];
		this.activations = new Matrix[this.num_hidden_layer + 1];
		this.activations[0] = new Matrix(this.num_input, 1, feature);
		for(int idl = 0; idl <= this.num_hidden_layer; ++idl) {
			this.interpretations[idl] = this.weights[idl].mult(
					this.activations[idl].appendAsVecBias());
			if(idl < this.num_hidden_layer) {
				this.activations[idl + 1] = this.interpretations[idl].relu();
			}
		}
		this.interpretations[this.num_hidden_layer].softmaxAsVecOnSelf();
	}
	
	/**
	 * backpropagation
	 * @param result
	 */
	public void backward(int result) {
		Matrix diff = new Matrix(this.num_output, 1, 
				this.interpretations[this.num_hidden_layer].getData());
		diff.addElement(result, 0, -1.0f);
		for(int idl = this.num_hidden_layer; idl >= 0 ; --idl) {
			if(idl < this.num_hidden_layer) {
				diff.multElemWiseOnSelf(this.interpretations[idl].relu_der());
			}
			Matrix delta = diff.ger(this.activations[idl].appendAsVecBias());
			if(idl > 0) {
				diff.conjAsVecOnSelf();
				diff = diff.mult(this.weights[idl]);
				diff = new Matrix(diff.getNumCol() - 1, 1, 
						Arrays.copyOf(diff.getData(), diff.getNumCol() - 1));
			}
			this.weights[idl].addOnSelf(delta, -this.stepsize);
		}
		
		++this.count_back_prop;
		if(this.count_back_prop >= this.count_unit) {
			this.stepsize /= 2.0f;
			this.count_back_prop = 0;
		}
	}
	
	/**
	 * return the result of classification, index of highest score
	 * @return
	 */
	public int getResult() {
		float max = -1.0f;
		int id_max = -1;
		float[] eval = this.interpretations[this.num_hidden_layer].getData();
		for(int i = 0; i < this.num_output; ++i) {
			if(max < eval[i]) {
				id_max = i;
				max = eval[i];
			}
		}
		return id_max;
	}
	
	/**
	 * return fake entropy, sum of square error between score and ideal score(01000 etc)
	 * @return
	 */
	public float getFakeEntropy() {
		float err = 0.0f;
		float[] eval = this.interpretations[this.num_hidden_layer].getData();
		for(int i = 0; i < this.num_output; ++i) {
			if(eval[i] < 0.5f) {
				err += eval[i] * eval[i];
			}
			else {
				err += (1.0f - eval[i]) * (1.0f - eval[i]);
			}
		}
		return err;
	}
}
