package ImageProc;

import java.util.Arrays;
import java.util.List;

/**
 * Created by tlou on 05.01.17.
 */
public class TableRecognizer
{
  private int[][] _hough_maxima_index;
  private float[] _hough_maxima_value;

  public void process()
  {
  }

  public void setHoughPoints(int[][] index, float[] value)
  {
    assert(index.length == value.length);
    final int length = index.length;
    this._hough_maxima_index = new int[length][2];
    this._hough_maxima_value = Arrays.copyOf(value, length);
    for(int i = 0; i < length; ++i)
    {
      assert(index[i].length == 2);
      this._hough_maxima_index[i][0] = index[i][0];
      this._hough_maxima_index[i][1] = index[i][1];
    }
//    for(int i = 0; i < length; ++i)
//      System.out.print(this._hough_maxima_index[i][0]+" ");
//    System.out.println();
//    for(int i = 0; i < length; ++i)
//      System.out.print(this._hough_maxima_index[i][1]+" ");
//    System.out.println();
  }

  public TableRecognizer()
  {
  }
}
