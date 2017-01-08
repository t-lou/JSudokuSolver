package ImageProc;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.Buffer;
import java.util.ArrayList;
import java.util.Arrays;

import javax.imageio.ImageIO;
import javax.swing.*;

import NN.Matrix;

public class ImageProc
{
  private Matrix _image;
  private Matrix _hough;
  private int _offset_hough_r;
  private Matrix _transform;
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
  public static void saveImage(BufferedImage image, String filename)
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

  private static BufferedImage drawHoughPoints(BufferedImage image, Color color, float[][] houghs)
  {
    final int color_val = color.getRGB();
    final int nr = image.getHeight();
    final int nc = image.getWidth();
    for(float[] hough : houghs)
    {
      final float cos_the = (float) Math.cos(Math.toRadians((double) hough[1]));
      final float sin_the = (float) Math.sin(Math.toRadians((double) hough[1]));
      final float r_val = hough[0];
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
    diff.erodeOnSelf(1, 0.2f);
    diff.setBoundary(0.0f);
    this._hough = this.transformHough(diff);
    this._hough.normalizeOnSelf();
    this._hough = this._hough.conv(this._gaussian_1_3, true);
    this._hough.normalizeOnSelf();
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
    }
  }

  public void drawTable(float[][][] lines, String filename)
  {
    BufferedImage image = ImageProc.matrixToImage(this._image, 1.0f);
    for(int id_dir = 0; id_dir < lines.length; ++id_dir)
    {
      assert(lines[id_dir].length == 4);
      ImageProc.drawHoughPoints(image, Color.red, lines[id_dir]);
    }
    ImageProc.saveImage(image, filename);
  }

  private static float[] getLineFromHough(float[] hough)
  {
    assert(hough.length == 2);
    double rad = Math.toRadians((double)hough[1]);
    return new float[]{(float)Math.cos(rad), (float)Math.sin(rad), -hough[0]};
  }

  private static float[] getIntersectionLines(float[] line0, float[] line1)
  {
    assert(line0.length == 3 && line1.length == 3);
    float[] intersection = new float[2];
    final float divisor = line0[1] * line1[0] - line0[0] * line1[1];
    intersection[1] = -(line0[2] * line1[0] - line0[0] * line1[2]) / divisor;
    intersection[0] = -(line0[1] * line1[2] - line0[2] * line1[1]) / divisor;
    return intersection;
  }

  private static float[] getCenter(float[][] points)
  {
    assert(points.length > 0);
    final int length = points[0].length;
    float[] center = new float[length];
    Arrays.fill(center, 0.0f);
    for(float[] point : points)
    {
      assert(point.length == length);
      for(int i = 0; i < length; ++i)
      {
        center[i] += point[i];
      }
    }
    for(int i = 0; i < length; ++i)
    {
      center[i] /= (float)length;
    }
    return center;
  }

  private static float getNorm(float[] vector)
  {
    float norm = 0.0f;
    for(float x : vector)
    {
      norm += x * x;
    }
    return (float)Math.sqrt((double)norm);
  }

  private static float[] getUnit(float[] vector)
  {
    final float norm = ImageProc.getNorm(vector);
    float[] unit = vector.clone();
    for(int i = 0; i < vector.length; ++i)
    {
      unit[i] /= norm;
    }
    return unit;
  }

  private static float[] diff(float[] vector0, float[] vector1)
  {
    assert(vector0.length == vector1.length);
    float[] d = new float[vector0.length];
    for(int i = 0; i < vector0.length; ++i)
    {
      d[i] = vector0[i] - vector1[i];
    }
    return d;
  }

  private static float[] rotate2D(float[] vector, float degree)
  {
    assert(vector.length == 2);
    final double radian = Math.toRadians((double)degree);
    final float c = (float)Math.cos(radian);
    final float s = (float)Math.sin(radian);
    return new float[]{c * vector[0] - s * vector[1], s * vector[0] + c * vector[1]};
  }

  private static float dot(float[] vector0, float[] vector1)
  {
    assert(vector0.length == vector1.length);
    float d = 0.0f;
    for(int i = 0; i < vector0.length; ++i)
    {
      d += vector0[i] * vector1[i];
    }
    return d;
  }

  private static float[][] getFourCorners(float[][][] hough_lines)
  {
    assert(hough_lines.length == 2);
    for(float[][] lines : hough_lines)
    {
      assert(lines.length == 4);
      for(float[] line : lines)
      {
        assert(line.length == 2);
      }
    }
    float corners[][] = new float[4][2];
    float lines[][] = new float[4][3];
    // the four corners are intersection of four lines with extreme r
    // suppose lines are already sorted by r value(translation)
    lines[0] = ImageProc.getLineFromHough(hough_lines[0][0]);
    lines[1] = ImageProc.getLineFromHough(hough_lines[1][0]);
    lines[2] = ImageProc.getLineFromHough(hough_lines[0][3]);
    lines[3] = ImageProc.getLineFromHough(hough_lines[1][3]);
    for(int i = 0; i < 4; ++i)
    {
      corners[i] = ImageProc.getIntersectionLines(lines[i], lines[(i + 1) % 4]);
    }
    // make sure corners are in specific order(for transform)
    float[] center = ImageProc.getCenter(corners);
    // get directions of corners, if after rotation of 90 degree directions[1] is very different, exchange
    if(ImageProc.dot(ImageProc.diff(corners[1], center),
        ImageProc.rotate2D(ImageProc.diff(corners[0], center), 90.0f)) < 0.0f)
    {
      // rotated corner #0 has opposite direction as corner #1
      float[] t = corners[1];
      corners[1] = corners[3];
      corners[3] = t;
    }
    return corners;
  }

  public void extractTransform(float[][][] hough_lines)
  {
    // transform from rectified image to real image
//    Matrix tf = new Matrix();
    final float[][] transformed = getFourCorners(hough_lines); // corners in one order
    // each block is 32x32, giving 2 pixel boundary (28x28 for classification)
    // total size is 288x288 (288 = 32 * 9)
    final float size = 288.0f;
    float[][] corners = new float[][]{
        {0.0f, 0.0f},
        {size, 0.0f},
        {size, size},
        {0.0f, size}};
    // solve with dlt (direct linear transform)
    // maybe normalization here
    float[] H_data = new float[72]; // 8x9 matrix
    Arrays.fill(H_data, 0.0f);
    for(int idp = 0; idp < 4; ++idp)
    {
      final int start = 18 * idp;
      H_data[start] = -corners[idp][0];
      H_data[start + 1] = -corners[idp][1];
      H_data[start + 2] = -1.0f;
      H_data[start + 6] = corners[idp][0] * transformed[idp][0];
      H_data[start + 7] = corners[idp][1] * transformed[idp][0];
      H_data[start + 8] = transformed[idp][0];
      H_data[start + 12] = -corners[idp][0];
      H_data[start + 13] = -corners[idp][1];
      H_data[start + 14] = -1.0f;
      H_data[start + 15] = corners[idp][0] * transformed[idp][1];
      H_data[start + 16] = corners[idp][1] * transformed[idp][1];
      H_data[start + 17] = transformed[idp][1];
    }
    float[] tf_data = new Matrix(8, 9, H_data).getZeroSpaceDoF1();
    this._transform = new Matrix(3, 3, tf_data);
  }

  public BufferedImage rectifyImage(int[] offset, int[] size)
  {
    assert(this._transform != null && offset.length == 2 && size.length == 2);
    final int[] size_image = new int[]{this._image.getNumRow(), this._image.getNumCol()};
    BufferedImage image = new BufferedImage(size[0], size[1], BufferedImage.TYPE_3BYTE_BGR);
    for(int r = 0; r < size[0]; ++r)
    {
      final float u = (float)(offset[0] + r);
      for(int c = 0; c < size[1]; ++c)
      {
        final float v = (float)(offset[1] + c);
        final Matrix pos = this._transform.multiply(new Matrix(3, 1, new float[]{v, u, 1.0f}));
        float[] pos_data = pos.getData();
        final int x = Math.round(pos_data[0] / pos_data[2]);
        final int y = Math.round(pos_data[1] / pos_data[2]);
        if(x >= 0 && x < size_image[1] && y >= 0 && y < size_image[0])
        {
          int val = (int)(this._image.getData()[x + size_image[1] * y] * 255.0f);
          image.setRGB(c, r, new Color(val, val, val).getRGB());
        }
      }
    }
    return image;
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
