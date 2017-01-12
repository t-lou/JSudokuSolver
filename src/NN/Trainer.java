package NN;

import ImageProc.ImageProc;

import java.util.Arrays;
import java.util.concurrent.ThreadLocalRandom;

public class Trainer
{
  private DataSource _data_train;
  private DataSource _data_valid;
  private Network _network;
  private int[] _dim;

  public Trainer(String path_train_image, String path_train_label,
                 String path_valid_image, String path_valid_label,
                 int num_hidden_layer, int num_hidden_unit, int num_output, float stepsize)
  {
    this._data_train = new DataSource(path_train_image, path_train_label);
    this._data_valid = new DataSource(path_valid_image, path_valid_label);
    this._network = new Network(num_hidden_layer, num_hidden_unit,
        this._data_train.getDim(), num_output, stepsize);
    this._dim = new int[]{this._data_train.getDim0(), this._data_train.getDim1()};
  }

  private static byte[] rotateClockwise(byte[] image, int row, int col)
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

  private byte[] permuteImage(byte[] image, int num_rot, boolean is_to_inverse)
  {
    byte[] result = Arrays.copyOf(image, image.length);
    int nr = this._dim[0];
    int nc = this._dim[1];
    for(int r = 0; r < num_rot; ++r)
    {
      result = Trainer.rotateClockwise(result, nr, nc);
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
  public void valid(int max_num_perm)
  {
    int count = 0;
    final int range_rand = 2 * max_num_perm;
    for(int i = 0; i < this._data_valid.getNumData(); ++i)
    {
      int perm_type = ThreadLocalRandom.current().nextInt(range_rand);
      this._network.forward(this.permuteImage(this._data_valid.getImage(i),
          perm_type % max_num_perm, perm_type < max_num_perm));
//      this._network.forward(this._data_valid.getImage(i));
      if(this._network.getResult() == this._data_valid.getLabel(i))
//      if(this._network.getResult() == (perm_type % 4)) // for training orientation
      {
        ++count;
      }
    }
    System.out.println(count + ":" + this._data_valid.getNumData());
  }

  /**
   * train on training data
   */
  public void train(int max_num_perm)
  {
    final int range_rand = 2 * max_num_perm;
    for(int i = 0; i < this._data_train.getNumData(); ++i)
    {
      int perm_type = ThreadLocalRandom.current().nextInt(range_rand);
      this._network.forward(this.permuteImage(this._data_train.getImage(i),
          perm_type % max_num_perm, perm_type < max_num_perm));
//      this._network.forward(this._data_train.getImage(i));
      this._network.backward(this._data_train.getLabel(i));
//      System.out.println(this._network.getResult() + " " + this._data_train.getLabel(i));
//      this._network.backward(perm_type % 4); // for training orientation
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
    return this._network;
  }

  /**
   * set trained network
   *
   * @return
   */
  public void SetNetwork(Network network)
  {
    this._network = network;
  }

  public boolean isNetworkNormal()
  {
    return this._network.isNetworkNormal();
  }
}
