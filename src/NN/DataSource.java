package NN;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;

public class DataSource
{
  private byte[][] _data_image;
  private int[] _data_label;
  private int _dim_in;
  private int _dim0, _dim1;

  /**
   * convert four bytes to int
   *
   * @param bytes
   * @return
   */
  private int bytesToInt(byte[] bytes)
  {
    ByteBuffer buffer = ByteBuffer.wrap(bytes);
    buffer.order(ByteOrder.BIG_ENDIAN);
    return buffer.getInt();
  }

  /**
   * convert array of (unsigned) byte to int(one byte to one int)
   *
   * @param bytes
   * @return
   */
  private int[] bytesToInts(byte[] bytes)
  {
    int[] result = new int[bytes.length];
    for(int i = 0; i < bytes.length; ++i)
    {
      result[i] = 0xFF & (int) bytes[i];
    }
    return result;
  }

//  /**
//   * convert (r*c) bytes to (r,c) float, each row normalized
//   *
//   * @param bytes
//   * @param len
//   * @param sub_len
//   * @return
//   */
//  private float[][] bytesToFloatArray(byte[] bytes, int len, int sub_len)
//  {
//    assert (len * sub_len == bytes.length);
//    float[][] result = new float[len][sub_len];
//    for(int id0 = 0; id0 < len; ++id0)
//    {
//      result[id0] = normalize(Arrays.copyOfRange(bytes, sub_len * id0,
//          sub_len * (id0 + 1)));
//    }
//    return result;
//  }

  /**
   * read file to bytes
   *
   * @param path
   * @return
   */
  private byte[] readFile(String path)
  {
    try
    {
      return Files.readAllBytes(Paths.get(path));
    } catch(Exception e)
    {
      return new byte[0];
    }
  }

  /**
   * load one pair of mnist files(one image and one label)
   *
   * @param path_image
   * @param path_label
   */
  public DataSource(String path_image, String path_label)
  {
    byte[] raw_image = this.readFile(path_image);
    byte[] raw_label = this.readFile(path_label);
    int num_image = bytesToInt(Arrays.copyOfRange(raw_image, 4, 8));
    int num_label = bytesToInt(Arrays.copyOfRange(raw_label, 4, 8));
    assert (num_image == num_label);
    this._dim0 = bytesToInt(Arrays.copyOfRange(raw_image, 8, 12));
    this._dim1 = bytesToInt(Arrays.copyOfRange(raw_image, 12, 16));
    this._dim_in = this._dim0 * this._dim1;
    this._data_label = bytesToInts(Arrays.copyOfRange(raw_label, 8, raw_label.length));
    this._data_image = new byte[this._data_label.length][this._dim_in];
    for(int i = 0, offset = 0; i < this._data_label.length; ++i, offset += this._dim_in)
    {
      System.arraycopy(raw_image, 16 + offset, this._data_image[i], 0, this._dim_in);
    }
  }

  /**
   * return id-th image
   *
   * @param id
   * @return
   */
  public byte[] getImage(int id)
  {
    assert (id < this._data_label.length);
    return this._data_image[id];
  }

  /**
   * return id-th label
   *
   * @param id
   * @return
   */
  public int getLabel(int id)
  {
    assert (id < this._data_label.length);
    return this._data_label[id];
  }

  /**
   * get size of dataset
   *
   * @return
   */
  public int getNumData()
  {
    return this._data_label.length;
  }

  /**
   * get dimensionality of feature(image)
   *
   * @return
   */
  public int getDim()
  {
    return this._dim_in;
  }

  public int getDim0()
  {
    return this._dim0;
  }

  public int getDim1()
  {
    return this._dim1;
  }
}
