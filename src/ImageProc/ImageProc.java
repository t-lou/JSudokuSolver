package ImageProc;

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.Arrays;

import javax.imageio.ImageIO;

import NN.Matrix;

public class ImageProc
{
  private Matrix image;
  private Matrix gaussian_1_3;
  private Matrix laplacian_4;
  private Matrix laplacian_8;
  private Matrix sobel_v;
  private Matrix sobel_h;
  private Matrix hough;
  private int offset_hough_r;

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

  private static BufferedImage drawHoughPoint(BufferedImage image, Color color, int r, int the, int offset_r)
  {
    final int color_val = color.getRGB();
    final float cos_the = (float)Math.cos(Math.toRadians((double)the));
    final float sin_the = (float)Math.sin(Math.toRadians((double)the));
    final float r_val = (float)(r + offset_r);
    final int nr = image.getHeight();
    final int nc = image.getWidth();
    if(Math.abs(cos_the) > Math.sqrt(0.5f))
    {
      for(int ir = 0; ir < nr; ++ir)
      {
        final int ic = Math.round((r_val - sin_the * (float)ir) / cos_the);
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
        final int ir = Math.round((r_val - cos_the * (float)ic) / sin_the);
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
    this.offset_hough_r = min_r;
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

  /**
   * some filters needed here
   */
  public void filter()
  {
    Matrix diff = this.image.conv(this.gaussian_1_3, true); // here filtered this.image
    diff = diff.conv(this.gaussian_1_3, true); // here filtered this.image
    // with sobel
    Matrix diff_v = diff.conv(this.sobel_v, false); // vertical diff of filtered image
    diff = diff.conv(this.sobel_h, false); // horizontal diff of filtered image
    diff.addAbsElemWiseOnSelf(diff_v); // omnidirectional diff of filtered image
    // with laplacian
//    diff = this.image.conv(this.laplacian_8, false);
//    diff.addAbsElemWiseOnSelf(diff);
    diff.normalizeOnSelf();
//    diff.thresholdOnSelf(0.3f);
    diff.erodeOnSelf(1, 0.2f);
//    diff = ImageProc.filterNonLocalMaxima(diff, 1, 0.8f); // only use local maxima for hough transform
//    ImageProc.saveImage(ImageProc.matrixToImage(diff, 1.0f), "D:\\home\\workspace\\tmp\\diff.png");
    diff.setBoundary(0.0f);
    this.hough = this.transformHough(diff);
    this.hough.normalizeOnSelf();
    this.hough = this.hough.conv(this.gaussian_1_3, true);
    this.hough.normalizeOnSelf();
//    ImageProc.saveImage(ImageProc.matrixToImage(this.hough, 1.0f), "D:\\home\\workspace\\tmp\\hough.png");
    int[][] index_max = this.hough.getLocalMaxima(4, 0.5f, true);
    float[] hough_local_max = this.hough.getElement(index_max);
    BufferedImage image = ImageProc.matrixToImage(this.image, 1.0f);
    for(int i = 0; i < index_max.length; ++i)
    {
      System.out.println(i + ": " + index_max[i][0] + " " + index_max[i][1]
          + ": " + hough_local_max[i]);
      image = ImageProc.drawHoughPoint(image, Color.red, index_max[i][0], index_max[i][1],
          this.offset_hough_r);
//      ImageProc.saveImage(image, "D:\\home\\workspace\\tmp\\" + i + ".png");
    }
//    ImageProc.saveImage(image, "D:\\home\\workspace\\tmp\\lines.png");
    System.out.println("there are "+ index_max.length + " local maximas");
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
    this.image = new Matrix(height, width, data);
  }

  public ImageProc()
  {
    this.gaussian_1_3 = new Matrix(3, 3, new float[]{
        0.077847f, 0.123317f, 0.077847f,
        0.123317f, 0.195346f, 0.123317f,
        0.077847f, 0.123317f, 0.077847f});

    this.laplacian_4 = new Matrix(3, 3, new float[]{
        0.0f, -1.0f, 0.0f, -1.0f, 4.0f, -1.0f, 0.0f, -1.0f, 0.0f});
    this.laplacian_8 = new Matrix(3, 3, new float[]{
        -1.0f, -1.0f,  -1.0f, -1.0f, 8.0f, -1.0f,  -1.0f, -1.0f, -1.0f});
//    this.laplacian_8 = new Matrix(3, 3, new float[]{
//        -0.5f, -1.0f,  -0.5f, -1.0f, 6.0f, -1.0f,  -0.5f, -1.0f, -0.5f});

//    this.sobel_v = new Matrix(3, 3, new float[]{
//        1.0f, 2.0f, 1.0f, 0.0f, 0.0f, 0.0f, -1.0f, -2.0f, -1.0f});
//    this.sobel_h = new Matrix(3, 3, new float[]{
//        1.0f, 0.0f, -1.0f, 2.0f, 0.0f, -2.0f, 1.0f, 0.0f, -1.0f});
    this.sobel_v = new Matrix(3, 3, new float[]{
        1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, -1.0f, -1.0f, -1.0f});
    this.sobel_h = new Matrix(3, 3, new float[]{
        1.0f, 0.0f, -1.0f, 1.0f, 0.0f, -1.0f, 1.0f, 0.0f, -1.0f});
  }
}
