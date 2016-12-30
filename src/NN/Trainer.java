package NN;

public class Trainer {
	private DataSource data_train;
	private DataSource data_valid;
	private Network network;
	
	public Trainer(String path_train_image, String path_train_label, 
			String path_valid_image, String path_valid_label,
			int num_hidden_layer, int num_hidden_unit, int num_output, float stepsize) {
		this.data_train = new DataSource(path_train_image, path_train_label);
		this.data_valid = new DataSource(path_valid_image, path_valid_label);
		this.network = new Network(num_hidden_layer, num_hidden_unit,
				this.data_train.getDim(), num_output, stepsize);
	}
	
	/**
	 * test on validation data
	 */
	public void valid() {
		int count = 0;
		for(int i = 0; i < this.data_valid.getNumData(); ++i)
		{
			this.network.forward(this.data_valid.getImage(i));
			if(this.network.getResult() == this.data_valid.getLabel(i)) {
				++count;
			}
		}
		System.out.println(count + ":" + this.data_valid.getNumData());
	}
	
	/**
	 * train on training data
	 */
	public void train() {
		for(int i = 0; i < this.data_train.getNumData(); ++i)
		{
			this.network.forward(this.data_train.getImage(i));
			this.network.backward(this.data_train.getLabel(i));
		}
		System.out.println("finished training");
	}
	
	/**
	 * return trained network
	 * @return
	 */
	public Network getNetwork() {
		return this.network;
	}
	
	/**
	 * set trained network
	 * @return
	 */
	public void SetNetwork(Network network) {
		this.network = network;
	}
}
