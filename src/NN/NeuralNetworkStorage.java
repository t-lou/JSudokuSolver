package NN;

import java.io.File;
import java.io.FileOutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;

/**
 * FILE FORMAT
 * 8 bytes "nnbytlou"
 * 4 bytes (1 int) number of layers, nl
 * nl*4 bytes (nl int) offset of each layer
 * for each layer:
 * 8 bytes (2 int) size (r,c)
 * r*c*4 bytes (r*c float) weights
 * <p>
 * BIGENDIAN
 */
public class NeuralNetworkStorage
{
  public NeuralNetworkStorage()
  {
  }

  /**
   * get representation on memory for i
   *
   * @param i
   * @return
   */
  private static byte[] intToBytes(int i)
  {
    return ByteBuffer.allocate(4).putInt(i).array();
  }

  /**
   * get int represented by bytes in memory
   *
   * @param bytes
   * @return
   */
  private static int bytesToInt(byte[] bytes)
  {
    ByteBuffer buffer = ByteBuffer.wrap(bytes);
    buffer.order(ByteOrder.BIG_ENDIAN);
    return buffer.getInt();
  }

  /**
   * write network to buffer
   *
   * @param network
   * @return
   */
  private static byte[] networkToBytes(Network network)
  {
    byte[] buffer;
    int count_byte = 8 + 4; // header
    int num_layer = network.getNumLayer();
    int[] offsets = new int[num_layer];
    int[][] sizes = new int[num_layer][2];

    count_byte += 4 * num_layer; // offsets
    for(int i = 0; i < num_layer; ++i)
    {
      offsets[i] = count_byte;
      count_byte += 8; // size
      sizes[i][0] = network.getLayer(i).getNumRow();
      sizes[i][1] = network.getLayer(i).getNumCol();
      count_byte += sizes[i][0] * sizes[i][1] * 4;
    }

    // create and write
    buffer = new byte[count_byte];
    System.arraycopy("nnbytlou".getBytes(), 0, buffer, 0, 8);
    // number of layer
    System.arraycopy(intToBytes(num_layer), 0, buffer, 8, 4);
    for(int i = 0; i < num_layer; ++i)
    {
      float[] data = network.getLayer(i).getData();
      ByteBuffer bf = ByteBuffer.allocate(4 * data.length);
      int address = 0;
      // offset
      System.arraycopy(intToBytes(offsets[i]), 0, buffer, 12 + 4 * i, 4);
      // sizes
      System.arraycopy(intToBytes(sizes[i][0]), 0, buffer, offsets[i], 4);
      System.arraycopy(intToBytes(sizes[i][1]), 0, buffer, offsets[i] + 4, 4);
      // weights
      for(float f : data)
      {
        bf.putFloat(address, f);
        address += 4;
      }
      System.arraycopy(bf.array(), 0, buffer, offsets[i] + 8, 4 * data.length);
    }

    return buffer;
  }

  /**
   * read network from buffer
   *
   * @param bytes
   * @return
   */
  private static Network bytesToNetwork(byte[] bytes)
  {
    Network network = new Network();
    assert ("nnbytlou".getBytes() == Arrays.copyOf(bytes, 8));
    int num_layer = bytesToInt(Arrays.copyOfRange(bytes, 8, 12));
    Matrix[] mats = new Matrix[num_layer];
    for(int idl = 0; idl < num_layer; ++idl)
    {
      int offset = bytesToInt(Arrays.copyOfRange(bytes, 12 + 4 * idl, 16 + 4 * idl));
      int num_row = bytesToInt(Arrays.copyOfRange(bytes, offset, offset + 4));
      int num_col = bytesToInt(Arrays.copyOfRange(bytes, offset + 4, offset + 8));
      float[] data = new float[num_row * num_col];
      ByteBuffer bf = ByteBuffer.wrap(Arrays.copyOfRange(bytes, offset + 8,
          offset + 8 + 4 * data.length));
      for(int id = 0; id < data.length; ++id)
      {
        data[id] = bf.getFloat(id * 4);
      }
      mats[idl] = new Matrix(num_row, num_col, data);
    }

    network.setLayers(mats);
    return network;
  }

  /**
   * save network to file
   *
   * @param filename
   * @param network
   */
  public static void save(String filename, Network network)
  {
    byte[] bytes = networkToBytes(network);
    try
    {
      File file = new File(filename);
      if(!file.exists())
      {
        file.createNewFile();
      }
      FileOutputStream fop = new FileOutputStream(file);
      fop.write(bytes);
      fop.flush();
      fop.close();
    } catch(Exception e)
    {
    }
  }

  /**
   * load network from file
   *
   * @param filename
   * @return
   */
  public static Network load(String filename)
  {
    Network network;
    try
    {
      network = bytesToNetwork(Files.readAllBytes(Paths.get(filename)));
    } catch(Exception e)
    {
      network = new Network();
    }
    return network;
  }
}
