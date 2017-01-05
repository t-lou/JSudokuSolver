package NN;

//import javax.imageio.ImageIO;
//import java.awt.*;
//import java.awt.image.BufferedImage;
//import java.io.File;

import java.util.Arrays;
import java.util.concurrent.ThreadLocalRandom;

public class Trainer
{
  private DataSource data_train;
  private DataSource data_valid;
  private Network network;
  private int[] dim;

  public Trainer(String path_train_image, String path_train_label,
                 String path_valid_image, String path_valid_label,
                 int num_hidden_layer, int num_hidden_unit, int num_output, float stepsize)
  {
    this.data_train = new DataSource(path_train_image, path_train_label);
    this.data_valid = new DataSource(path_valid_image, path_valid_label);
    this.network = new Network(num_hidden_layer, num_hidden_unit,
        this.data_train.getDim(), num_output, stepsize);
    this.dim = new int[]{this.data_train.getDim0(), this.data_train.getDim1()};
  }

  private byte[] rotateClockwise(byte[] image, int row, int col)
  {
    assert (row * col == image.length);
    byte[] tmp = new byte[image.length];

    for(int r = 0; r < row; ++r)
    {
      final int c1 = row - 1 - r;
      final int row_start = r * col;
      for(int c = 0; c < col; ++c)
      {
        tmp[c * row + c1] = image[row_start + c];
      }
    }
    return tmp;
  }

//  private int count;
//  private void saveImage(byte[] bytes)
//  {
//    BufferedImage image = new BufferedImage(this.data_train.getDim0(),
//        this.data_train.getDim1(), BufferedImage.TYPE_3BYTE_BGR);
//    for(int r = 0; r < this.data_train.getDim0(); ++r)
//    {
//      int row_start = this.data_train.getDim1() * r;
//      for(int c = 0; c < this.data_train.getDim1(); ++c)
//      {
//        int val = (0xFF & bytes[c + row_start]);
//        image.setRGB(c, r, new Color(val, val, val).getRGB());
//      }
//    }
//    try
//    {
//      File ouptut = new File("D:\\home\\workspace\\tmp\\"+this.count+".png");
//      ImageIO.write(image, "png", ouptut);
//      ++this.count;
//    } catch(Exception e)
//    {
//    }
//  }

  private byte[] permuteImage(byte[] image, int type)
  {
    final boolean is_to_inverse = type < 4;
    final int num_rot = type % 4;
    byte[] result = Arrays.copyOf(image, image.length);
    int nr = this.dim[0];
    int nc = this.dim[1];
    for(int r = 0; r < num_rot; ++r)
    {
      result = this.rotateClockwise(result, nr, nc);
      int t = nr;
      nr = nc;
      nc = t;
    }
    if(is_to_inverse)
    {
      int len = result.length;
      for(int i = 0; i < len; ++i)
      {
        result[i] = (byte) (0xFF & (0xFF - (0xFFFFFFFF & (int) result[i])));
      }
    }
//    this.saveImage(result);
    return result;
  }

  /**
   * test on validation data
   */
  public void valid()
  {
    int count = 0;
    for(int i = 0; i < this.data_valid.getNumData(); ++i)
    {
      int perm_type = ThreadLocalRandom.current().nextInt(0, 8);
      this.network.forward(this.permuteImage(this.data_train.getImage(i), perm_type));
      if(this.network.getResult() == this.data_valid.getLabel(i))
//      if(this.network.getResult() == (perm_type % 4)) // for training orientation
      {
        ++count;
      }
    }
    System.out.println(count + ":" + this.data_valid.getNumData());
  }

  /**
   * train on training data
   */
  public void train()
  {
    for(int i = 0; i < this.data_train.getNumData(); ++i)
    {
      int perm_type = ThreadLocalRandom.current().nextInt(0, 8);
      this.network.forward(this.permuteImage(this.data_train.getImage(i), perm_type));
      this.network.backward(this.data_train.getLabel(i));
//      this.network.backward(perm_type % 4); // for training orientation
    }
    System.out.println("finished training");
  }

  /**
   * return trained network
   *
   * @return
   */
  public Network getNetwork()
  {
    return this.network;
  }

  /**
   * set trained network
   *
   * @return
   */
  public void SetNetwork(Network network)
  {
    this.network = network;
  }
}
