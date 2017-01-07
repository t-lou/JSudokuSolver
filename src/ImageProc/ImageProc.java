package ImageProc;

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.Arrays;

import javax.imageio.ImageIO;

import NN.Matrix;

public class ImageProc
{
  private Matrix _image;
  private Matrix _hough;
  private int _offset_hough_r;
  private static Matrix _gaussian_1_3 = new Matrix(3, 3, new float[]{
      0.077847f, 0.123317f, 0.077847f,
      0.123317f, 0.195346f, 0.123317f,
      0.077847f, 0.123317f, 0.077847f});
  private static Matrix _laplacian_4 = new Matrix(3, 3, new float[]{
      0.0f, -1.0f, 0.0f, -1.0f, 4.0f, -1.0f, 0.0f, -1.0f, 0.0f});
  private static Matrix _laplacian_8 = new Matrix(3, 3, new float[]{
      -1.0f, -1.0f, -1.0f, -1.0f, 8.0f, -1.0f, -1.0f, -1.0f, -1.0f});
  private static Matrix _sobel_v = new Matrix(3, 3, new float[]{
      1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, -1.0f, -1.0f, -1.0f});
  private static Matrix _sobel_h = new Matrix(3, 3, new float[]{
      1.0f, 0.0f, -1.0f, 1.0f, 0.0f, -1.0f, 1.0f, 0.0f, -1.0f});
  private int[][] _hough_maxima_index;
  private float[] _hough_maxima_value;

  /**
   * save image to file for debugging
   *
   * @param image
   * @param filename
   */
  private static void saveImage(BufferedImage image, String filename)
  {
    try
    {
      File ouptut = new File(filename);
      ImageIO.write(image, "png", ouptut);
    } catch(Exception e)
    {
    }
  }

  private static BufferedImage matrixToImage(Matrix matrix, float scale)
  {
    BufferedImage image = new BufferedImage(matrix.getNumCol(),
        matrix.getNumRow(), BufferedImage.TYPE_3BYTE_BGR);
    for(int r = 0; r < matrix.getNumRow(); ++r)
    {
      final int row_start = matrix.getNumCol() * r;
      for(int c = 0; c < matrix.getNumCol(); ++c)
      {
        int val = (int) (255.0f * matrix.getData()[row_start + c] * scale);
        if(val > 255)
        {
          val = 255;
        }
        else if(val < 0)
        {
          val = 0;
        }
        image.setRGB(c, r, new Color(val, val, val).getRGB());
      }
    }
    return image;
  }

  private static BufferedImage drawHoughPoint(BufferedImage image, Color color, int r, int the)
  {
    final int color_val = color.getRGB();
    final float cos_the = (float) Math.cos(Math.toRadians((double) the));
    final float sin_the = (float) Math.sin(Math.toRadians((double) the));
    final float r_val = (float) r;
    final int nr = image.getHeight();
    final int nc = image.getWidth();
    if(Math.abs(cos_the) > Math.sqrt(0.5f))
    {
      for(int ir = 0; ir < nr; ++ir)
      {
        final int ic = Math.round((r_val - sin_the * (float) ir) / cos_the);
        if(ic >= 0 && ic < nc)
        {
          image.setRGB(ic, ir, color_val);
        }
      }
    }
    else
    {
      for(int ic = 0; ic < nc; ++ic)
      {
        final int ir = Math.round((r_val - cos_the * (float) ic) / sin_the);
        if(ir >= 0 && ir < nr)
        {
          image.setRGB(ic, ir, color_val);
        }
      }
    }
    return image;
  }

  /**
   * return hough transform
   *
   * @param image
   * @return
   */
  public Matrix transformHough(Matrix image)
  {
    float[] coss = new float[180];
    float[] sins = new float[180];
    for(int the = 0; the < 180; ++the)
    {
      double rad = Math.toRadians((double) the);
      coss[the] = (float) Math.cos(rad);
      sins[the] = (float) Math.sin(rad);
    }

    int max_r = 0;
    int min_r = 0;
    final int num_row = image.getNumRow();
    final int num_col = image.getNumCol();
    for(int row = 0; row < num_row; row += 5)
    {
      for(int col = 0; col < num_col; col += 5)
      {
        for(int the = 0; the < 180; the += 10)
        {
          final int rv = Math.round((float) col * coss[the] + (float) row * sins[the]);
          if(max_r < rv)
          {
            max_r = rv;
          }
          if(min_r > rv)
          {
            min_r = rv;
          }
        }
      }
    }

    float[] hough_values = new float[(max_r - min_r) * 180];
    Arrays.fill(hough_values, 0.0f);

    int index = 0;
    for(int row = 0; row < num_row; ++row)
    {
      for(int col = 0; col < num_col; ++col)
      {
        final float val = image.getData()[index];
        for(int the = 0; the < 180; ++the)
        {
          int rv = Math.round((float) col * coss[the] + (float) row * sins[the]);
          if(rv < max_r && rv > min_r)
          {
            hough_values[(rv - min_r) * 180 + the] += val;
          }
        }
        ++index;
      }
    }
    this._offset_hough_r = min_r;
    return new Matrix(max_r - min_r, 180, hough_values);
  }

  private static Matrix filterNonLocalMaxima(Matrix image, int radius, float minimal)
  {
    final int nr = image.getNumRow(), nc = image.getNumCol();
    final int r_max = nr - radius, c_max = nc - radius;
    int[][] index_max = image.getLocalMaxima(radius, minimal, false);
    float[] filtered = new float[nr * nc];
    float[] original = image.getData();
    Arrays.fill(filtered, 0.0f);
    for(int[] index : index_max)
    {
      if(index[0] >= radius && index[0] < r_max && index[1] >= radius && index[1] < c_max)
      {
        final float thres = original[index[0] * nc + index[1]] * 0.8f;
        for(int r = -radius; r <= radius; ++r)
        {
          final int row_start = (r + index[0]) * nc;
          for(int c = -radius; c <= radius; ++c)
          {
            final int index_1d = row_start + c + index[1];
            if(original[index_1d] > thres)
            {
              filtered[index_1d] = original[index_1d];
            }
          }
        }
      }
    }
    return new Matrix(nr, nc, filtered);
  }

  private static Matrix detectEdge(Matrix image)
  {
    final int num_row = image.getNumRow();
    final int num_col = image.getNumCol();
    final int length = num_row * num_col;
    final float[] diff_h = image.conv(ImageProc._sobel_h, false).getData();
    final float[] diff_v = image.conv(ImageProc._sobel_v, false).getData();
    float[] diff = new float[length];
    for(int i = 0; i < length; ++i)
    {
      diff[i] = (float)Math.sqrt(diff_h[i] * diff_h[i] + diff_v[i] * diff_v[i]);
    }
    return new Matrix(num_row, num_col, diff);
  }

  /**
   * some filters needed here
   */
  public void extractLines()
  {
    Matrix diff = ImageProc.detectEdge(this._image.conv(this._gaussian_1_3, true));
    diff.normalizeOnSelf();
//    ImageProc.saveImage(ImageProc.matrixToImage(diff, 1.0f), "/tmp/tmp/diff.png");
//    ImageProc.saveImage(ImageProc.matrixToImage(diff, 1.0f), "D:\\home\\workspace\\xtmp\\diff.png");
    diff.erodeOnSelf(1, 0.2f);
//    ImageProc.saveImage(ImageProc.matrixToImage(diff, 1.0f), "/tmp/tmp/erode.png");
//    ImageProc.saveImage(ImageProc.matrixToImage(diff, 1.0f), "D:\\home\\workspace\\tmp\\erode.png");
    diff.setBoundary(0.0f);
    this._hough = this.transformHough(diff);
    this._hough.normalizeOnSelf();
    this._hough = this._hough.conv(this._gaussian_1_3, true);
    this._hough.normalizeOnSelf();
//    ImageProc.saveImage(ImageProc.matrixToImage(this.hough, 1.0f), "/tmp/tmp/hough.png");
//    ImageProc.saveImage(ImageProc.matrixToImage(this.hough, 1.0f), "D:\\home\\workspace\\tmp\\hough.png");
    this._hough_maxima_index = this._hough.getLocalMaxima(10, 0.6f, true);
    this._hough_maxima_value = this._hough.getElement(this._hough_maxima_index);
    final int length = this._hough_maxima_index.length;
//    BufferedImage image = ImageProc.matrixToImage(this._image, 1.0f);
    for(int i = 0; i < length; ++i)
    {
      this._hough_maxima_index[i][0] += this._offset_hough_r;
      // correct negative r
      if(this._hough_maxima_index[i][0] < 0)
      {
        this._hough_maxima_index[i][1] -= 180;
        this._hough_maxima_index[i][0] = -this._hough_maxima_index[i][0];
      }
//      System.out.println(i + ": " + index_max[i][0] + " " + index_max[i][1]
//          + ": " + hough_local_max[i]);
//      image = ImageProc.drawHoughPoint(image, Color.red, index_max[i][0], index_max[i][1]);
//      ImageProc.saveImage(image, "/tmp/tmp/" + i + ".png");
//      ImageProc.saveImage(image, "D:\\home\\workspace\\tmp\\" + i + ".png");
    }
//    ImageProc.saveImage(image, "/tmp/tmp/lines.png");
//    ImageProc.saveImage(image, "D:\\home\\workspace\\tmp\\lines.png");
//    System.out.println("there are " + index_max.length + " local maximas");
  }

  public void drawTable(float[][][] lines, String filename)
  {
    BufferedImage image = ImageProc.matrixToImage(this._image, 1.0f);
    for(int id_dir = 0; id_dir < lines.length; ++id_dir)
    {
      assert(lines[id_dir].length == 4);
      for(int id_line = 0; id_line < lines[id_dir].length; ++id_line)
      {
        ImageProc.drawHoughPoint(image, Color.red,
            Math.round(lines[id_dir][id_line][0]), Math.round(lines[id_dir][id_line][1]));
      }
    }
    ImageProc.saveImage(image, filename);
  }

  /**
   * set input image
   *
   * @param image
   */
  public void setImage(BufferedImage image)
  {
    final int height = image.getHeight();
    final int width = image.getWidth();
    float[] data = new float[height * width];
    for(int r = 0, row_start = 0; r < height; ++r, row_start += width)
    {
      for(int c = 0; c < width; ++c)
      {
        Color color = new Color(image.getRGB(c, r));
        data[row_start + c] = (0.299f * (float) color.getRed()
            + 0.114f * (float) color.getBlue() + 0.587f * (float) color.getGreen()) / 255.0f;
      }
    }
    this._image = new Matrix(height, width, data);
  }

  public float[] getHoughMaximaValue()
  {
    return this._hough_maxima_value;
  }

  public int[][] getHoughMaximaIndex()
  {
    return this._hough_maxima_index;
  }

  public ImageProc()
  {
  }
}
