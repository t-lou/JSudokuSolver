package NN;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Created by tlou on 15.01.17.
 */
public class NetworkGroup
{
  private List<Network> _networks;

  public boolean setNetworkByFile(String[] filenames)
  {
    for(String filename : filenames)
    {
      if(!new File(filename).exists())
      {
        this._networks.clear();
        return false;
      }
      this._networks.add(NeuralNetworkStorage.load(filename));
    }
    return true;
  }

  public void forward(byte[] feature)
  {
    for(Network nn : this._networks)
    {
      nn.forward(feature);
    }
  }

  public int getResult()
  {
    return this.getResult(0.0f);
  }

  public int getResult(float min_score)
  {
    assert(!this._networks.isEmpty());
    final int length = this._networks.get(0).getDemensionality()[1];
    float[] likelihood = new float[length];
    Arrays.fill(likelihood, 0.0f);
    for(Network nn : this._networks)
    {
      float[] ll = nn.getLikelihood();
      for(int i = 0; i < length; ++i)
      {
        likelihood[i] += ll[i];
      }
    }
    return Network.getResult(likelihood, min_score);
  }

  public NetworkGroup()
  {
    this._networks = new ArrayList<Network>();
  }
}
