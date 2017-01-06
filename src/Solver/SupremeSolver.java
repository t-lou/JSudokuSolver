package Solver;

import java.io.File;
import javax.imageio.ImageIO;

import ImageProc.ImageProc;

public class SupremeSolver
{
  private SudokuSolver _solver;
  private ImageProc _img_proc;

  public void preproc()
  {
    this._img_proc.filter();
  }

  public SupremeSolver(String filename)
  {
    this._img_proc = new ImageProc();
    try
    {
      this._img_proc.setImage(ImageIO.read(new File(filename)));
    } catch(Exception e)
    {
    }
  }
}
