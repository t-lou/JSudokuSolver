package NN;

import java.util.Arrays;
import java.util.Random;

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
    assert(row * col == image.length);
    byte[] tmp = new byte[image.length];

    for(int r = 0; r < row; ++r)
    {
      int c1 = row - 1 - r;
      int row_start = r * col;
      for(int c = 0; c < col; ++c)
      {
        tmp[c * row + c1] = image[row_start + c];
      }
    }
    return tmp;
  }

  private byte[] permuteImage(byte[] image, int type)
  {
    boolean is_to_inverse = type < 4;
    int num_rot = type % 4;
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
    return result;
  }

  /**
   * test on validation data
   */
  public void valid()
  {
    int count = 0;
    Random rand = new Random();
    for(int i = 0; i < this.data_valid.getNumData(); ++i)
    {
//			this.network.forward(this.data_valid.getImage(i));
      this.network.forward(this.permuteImage(this.data_valid.getImage(i),
          rand.nextInt() % 8));
      if(this.network.getResult() == this.data_valid.getLabel(i))
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
    Random rand = new Random();
    for(int i = 0; i < this.data_train.getNumData(); ++i)
    {
//			this.network.forward(this.data_train.getImage(i));
      this.network.forward(this.permuteImage(this.data_train.getImage(i),
          rand.nextInt() % 8));
      this.network.backward(this.data_train.getLabel(i));
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
