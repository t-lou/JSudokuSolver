package Solver;

import java.io.File;
import javax.imageio.ImageIO;

import ImageProc.ImageProc;
import ImageProc.TableRecognizer;

public class SupremeSolver
{
  private SudokuSolver _solver;
  private TableRecognizer _table_recognizer;
  private ImageProc _img_proc;

  public void process()
  {
    this._img_proc.extractLines();
    this._table_recognizer.setHoughPoints(this._img_proc.getHoughMaximaIndex(),
        this._img_proc.getHoughMaximaValue());
    this._table_recognizer.proceed();
//    this._img_proc.drawTable(this._table_recognizer.getTableLines(), "/tmp/lines.png");
  }

  public SupremeSolver(String filename)
  {
    this._img_proc = new ImageProc();
    this._table_recognizer = new TableRecognizer();
    try
    {
      this._img_proc.setImage(ImageIO.read(new File(filename)));
    } catch(Exception e)
    {
    }
  }
}
