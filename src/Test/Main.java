package Test;

import NN.Matrix;
import NN.NeuralNetworkStorage;
import NN.Trainer;
import Solver.SudokuSolver;
import Solver.SupremeSolver;

public class Main
{
  public static void main(String[] args)
  {
    System.out.println("Started");
    long start = System.currentTimeMillis();

    // test digit recognition
//    String path = "D:\\home\\workspace\\mnist\\";
//    String path = "/home/tlou/workspace/mnist/";
//    String path_train_image = path + "train-images-idx3-ubyte";
//    String path_train_label = path + "train-labels-idx1-ubyte";
//    String path_valid_image = path + "t10k-images-idx3-ubyte";
//    String path_valid_label = path + "t10k-labels-idx1-ubyte";
//    String path_train_image = path + "notMNIST_large_image";
//    String path_train_label = path + "notMNIST_large_label";
//    String path_valid_image = path + "notMNIST_small_image";
//    String path_valid_label = path + "notMNIST_small_label";
//    Trainer trainer = new Trainer(path_train_image, path_train_label,
//        path_valid_image, path_valid_label, 4, 400, 10, 0.0001f);
//    for(int i = 0; i < 40; ++i)
//    {
//      System.out.println("round " + i);
//      trainer.train();
//      trainer.valid();
//    }
//    NeuralNetworkStorage.save("/home/tlou/nn.nn", trainer.getNetwork());
//    trainer.SetNetwork(NeuralNetworkStorage.load("/home/tlou/nn.nn"));
//    System.out.println("loaded network");
//    trainer.valid();

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
//		SudokuSolver solver = new SudokuSolver(mapping);
//		solver.solve();

//		// test total solver
    SupremeSolver ssolver = new SupremeSolver("/home/tlou/Downloads/cbhsudoku.png");
//		SupremeSolver ssolver = new SupremeSolver("D:\\home\\workspace\\cbhsudoku.png");
//    ssolver.process();

//    Matrix mat = new Matrix(3, 3, new float[]{
//        1.0f, 2.0f, 0.0f, 2.0f, 4.0f, 1.0f, 2.0f, 1.0f, 0.0f
//    });
//    mat.disp();
//    mat.inv().disp();
//    mat.eliminateGaussJordanOnSelf();
//    mat.disp();

    long end = System.currentTimeMillis();
    System.out.println("Took " + 0.001f * (float) (end - start) + " s");
  }
}
