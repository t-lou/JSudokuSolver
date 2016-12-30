package Solver;

import NN.Trainer;

public class Main {    
	public static void main(String[] args) {
		System.out.println("Started");
		
		// test digit recognition
		String path_train_image = "/home/tlou/Downloads/train-images-idx3-ubyte";
		String path_train_label = "/home/tlou/Downloads/train-labels-idx1-ubyte";
		String path_valid_image = "/home/tlou/Downloads/t10k-images-idx3-ubyte";
		String path_valid_label = "/home/tlou/Downloads/t10k-labels-idx1-ubyte";
		Trainer trainer = new Trainer(path_train_image, path_train_label, 
				path_valid_image, path_valid_label, 1, 300, 10, 0.00002f);
		// trainer.train();
		// trainer.valid();
		
		// test sudoku solver
		int[][] mapping = new int[][]{
			{0, 0, 5},
			{0, 1, 3},
			{0, 4, 7},
			{1, 0, 6},
			{1, 3, 1},
			{1, 4, 9},
			{1, 5, 5},
			{2, 1, 9},
			{2, 2, 8},
			{2, 7, 6},
			{3, 0, 8},
			{3, 4, 6},
			{3, 8, 3},
			{4, 0, 4},
			{4, 3, 8},
			{4, 5, 3},
			{4, 8, 1},
			{5, 0, 7},
			{5, 4, 2},
			{5, 8, 6},
			{6, 1, 6},
			{6, 6, 2},
			{6, 7, 8},
			{7, 3, 4},
			{7, 4, 1},
			{7, 5, 9},
			{7, 8, 5},
			{8, 4, 8},
			{8, 7, 7},
			{8, 8, 9}};
		Solver solver = new Solver(mapping);
		solver.solve();
		
		// test total solver
		// SupremeSolver solver = new SupremeSolver("/home/tlou/Downloads/cbhsudoku.png");
    }
}
