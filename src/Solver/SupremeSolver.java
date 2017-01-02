package Solver;

import java.io.File;
import javax.imageio.ImageIO;

import ImageProc.ImageProc;

public class SupremeSolver
{
  private SudokuSolver solver;
  private ImageProc img_proc;

  public void preproc()
  {
    this.img_proc.filter();
  }

  public SupremeSolver(String filename)
  {
    this.img_proc = new ImageProc();
    try
    {
      this.img_proc.setImage(ImageIO.read(new File(filename)));
    } catch (Exception e)
    {
    }
  }
}
