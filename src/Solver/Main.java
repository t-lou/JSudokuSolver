package Solver;

import NN.Trainer;

public class Main {    
	public static void main(String[] args) {
		SupremeSolver solver = new SupremeSolver("/home/tlou/Downloads/cbhsudoku.png");
		System.out.println("Started");
		
		String path_train_image = "/home/tlou/Downloads/train-images-idx3-ubyte";
		String path_train_label = "/home/tlou/Downloads/train-labels-idx1-ubyte";
		String path_valid_image = "/home/tlou/Downloads/t10k-images-idx3-ubyte";
		String path_valid_label = "/home/tlou/Downloads/t10k-labels-idx1-ubyte";
		Trainer trainer = new Trainer(path_train_image, path_train_label, 
				path_valid_image, path_valid_label, 2, 300, 10, 0.00002f);
		trainer.train();
		trainer.train();
		trainer.train();
		trainer.train();
		trainer.train();
		trainer.train();
		trainer.valid();
    }
}
