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
      int row_start = matrix.getNumCol() * r;
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
    int color_val = color.getRGB();
    float cos_the = (float)Math.cos(Math.toRadians((double)the));
    float sin_the = (float)Math.sin(Math.toRadians((double)the));
    float r_val = (float)(r + offset_r);
    int nr = image.getHeight();
    int nc = image.getWidth();
    System.out.print(cos_the+" "+sin_the);
    if(Math.abs(cos_the) > Math.sqrt(0.5f))
    {
      System.out.println(" one");
      for(int ir = 0; ir < nr; ++ir)
      {
        int ic = Math.round((r_val - sin_the * (float)ir) / cos_the);
        if(ic >= 0 && ic < nc)
        {
          image.setRGB(ic, ir, color_val);
        }
      }
    }
    else
    {
      System.out.println(" another");
      for(int ic = 0; ic < nc; ++ic)
      {
        int ir = Math.round((r_val - cos_the * (float)ic) / sin_the);
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
    int num_row = image.getNumRow();
    int num_col = image.getNumCol();
    for(int row = 0; row < num_row; row += 5)
    {
      for(int col = 0; col < num_col; col += 5)
      {
        for(int the = 0; the < 180; the += 10)
        {
          int rv = Math.round((float) col * coss[the] + (float) row * sins[the]);
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
        float val = image.getData()[index];
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

  /**
   * some filters needed here
   */
  public void filter()
  {
    Matrix diff = this.image.conv(this.gaussian_1_3, true); // here filtered this.image
    Matrix diff_v = diff.conv(this.sobel_v, false); // vertical diff of filtered imageimage
    diff = diff.conv(this.sobel_h, false); // horizonal diff of filtered imageimage
    diff.addAbsElemWiseOnSelf(diff_v); // omnidirectional diff of filtered image
    diff.setBoundary(0.0f);
    this.hough = this.transformHough(diff);
    this.hough.normalizeOnSelf();
    this.hough = this.hough.conv(this.gaussian_1_3, true);
    this.hough.normalizeOnSelf();
    int[][] index_max = this.hough.getLocalMaxima();
    float[] hough_local_max = this.hough.getElement(index_max);
    BufferedImage image = ImageProc.matrixToImage(this.image, 1.0f);
    for(int i = 0; i < index_max.length; ++i)
    {
      System.out.println(i + ": " + index_max[i][0] + " " + index_max[i][1]
          + ": " + hough_local_max[i]);
      image = ImageProc.drawHoughPoint(image, new Color(255, 0, 0), index_max[i][0], index_max[i][1],
          this.offset_hough_r);
//      ImageProc.saveImage(image, "D:\\home\\workspace\\tmp\\" + i + ".png");
    }
    System.out.println("there are "+ index_max.length + " local maximas");
  }

  /**
   * set input image
   *
   * @param image
   */
  public void setImage(BufferedImage image)
  {
    int height = image.getHeight();
    int width = image.getWidth();
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

    this.sobel_v = new Matrix(3, 3, new float[]{
        1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, -1.0f, -1.0f, -1.0f});
    // sobel vertical
    this.sobel_h = new Matrix(3, 3, new float[]{
        1.0f, 0.0f, -1.0f, 1.0f, 0.0f, -1.0f, 1.0f, 0.0f, -1.0f});
  }
}
