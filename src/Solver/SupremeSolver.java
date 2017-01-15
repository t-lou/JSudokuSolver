package Solver;

import java.awt.image.BufferedImage;
import java.io.File;
import javax.imageio.ImageIO;

import ImageProc.ImageProc;
import ImageProc.TableRecognizer;
import NN.Network;
import NN.NetworkGroup;
import NN.NeuralNetworkStorage;

public class SupremeSolver
{
  private SudokuSolver _solver;
  private TableRecognizer _table_recognizer;
  private ImageProc _img_proc;
//  private Network _classifier_digit;
  private NetworkGroup _classifier_digit;
  private Network _classifier_orientation;

  public void process()
  {
    assert(this._classifier_digit != null && this._classifier_orientation != null);
    this._img_proc.extractLines();
    this._table_recognizer.setHoughPoints(this._img_proc.getHoughMaximaIndex(),
        this._img_proc.getHoughMaximaValue());
    this._table_recognizer.proceed();
    this._img_proc.extractTransform(this._table_recognizer.getTableLines());

    for(int i = 0; i < 9; ++i)
    {
      for(int j = 0; j < 9; ++j)
      {
        byte[] feature = this._img_proc.getBlock(i, j);
        ImageProc.saveImage(feature, 28, 28, "/tmp/"+i+""+j+".png");
        this._classifier_orientation.forward(feature);
        this._classifier_digit.forward(feature);
        System.out.println(i + " " + j + ": " + this._classifier_digit.getResult(0.2f)
          + " " + this._classifier_orientation.getResult(0.2f));
      }
    }
  }

//  public void loadClassifierDigit(String filename)
//  {
//    this._classifier_digit = NeuralNetworkStorage.load(filename);
//  }
  public void loadClassifierDigit(String[] filenames)
  {
    this._classifier_digit.setNetworkByFile(filenames);
  }

  public void loadClassifierOrientation(String filename)
  {
    this._classifier_orientation = NeuralNetworkStorage.load(filename);
  }

  public void setImage(BufferedImage image)
  {
    this._img_proc.setImage(image);
  }

  public void setImage(String filename)
  {
    try
    {
      this.setImage(ImageIO.read(new File(filename)));
    } catch(Exception e)
    {}
  }

  public SupremeSolver()
  {
    this._img_proc = new ImageProc();
    this._table_recognizer = new TableRecognizer();
    this._classifier_digit = new NetworkGroup();
  }
}
