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
    this._img_proc.extractTransform(this._table_recognizer.getTableLines());
//    ImageProc.saveImage(this._img_proc.rectifyImage(new int[]{0, 0}, new int[]{288, 288}), "/tmp/rect.png");
    for(int i = 0; i < 9; ++i)
      for(int j = 0; j < 9; ++j)
        ImageProc.saveImage(this._img_proc.rectifyBlock(i, j), "/tmp/rect"+i+""+j+".png");
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
