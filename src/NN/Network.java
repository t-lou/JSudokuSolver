package NN;

import java.util.Arrays;

public class Network
{
  private int _num_hidden_layer;
  private int _num_input;
  private int _num_output;
  private float _stepsize;
  private float _stepsize_base;
  private int _count_back_prop;
  private int _num_iter_to_damp_stepsize;
  private Matrix[] _weights;
  private Matrix[] _interpretations;
  private Matrix[] _activations;

  public Network()
  {
  }

  public Network(int num_hidden_layer, int num_hidden_unit,
                 int num_input, int num_output, float stepsize)
  {
    this._num_hidden_layer = num_hidden_layer;
    this._num_input = num_input;
    this._num_output = num_output;
    this._weights = new Matrix[num_hidden_layer + 1];
    this._stepsize_base = stepsize;
    this._stepsize = this._stepsize_base;
    this._count_back_prop = 0;

    // last row is for bias
    this._weights[0] = new Matrix(num_hidden_unit, num_input + 1, true);
    for(int i = 1; i < num_hidden_layer; ++i)
    {
      this._weights[i] = new Matrix(num_hidden_unit, num_hidden_unit + 1, true);
    }
    this._weights[num_hidden_layer] = new Matrix(num_output, num_hidden_unit + 1, true);
    this._num_iter_to_damp_stepsize = 0;
    for(Matrix w : this._weights)
    {
      this._num_iter_to_damp_stepsize += w.getNumCol() * w.getNumRow();
    }
    this._num_iter_to_damp_stepsize *= 2;
  }

  /**
   * convert (unsigned) bytes to floats(one byte to one float), then normalize the array
   *
   * @param bytes
   * @return
   */
  private float[] normalize(byte[] bytes)
  {
    float[] result = new float[bytes.length];
    float sum = 0.0f;
    float sum_sq = 0.0f;
    float mean;

    for(int id = 0; id < bytes.length; ++id)
    {
      int val_i = 0xFF & (int) bytes[id];
      float val = 1.0f - ((float) val_i) / 255.0f;
      sum += val;
      sum_sq += val * val;
      result[id] = val;
    }

    mean = sum / (float) bytes.length;
    final float dev = (float) Math.sqrt(sum_sq / (float) bytes.length - mean * mean);

    if(dev > 0.0001f)
    {
      for(int id = 0; id < bytes.length; ++id)
      {
        result[id] = (result[id] - mean) / dev;
      }
    }
    else
    {
      for(int id = 0; id < bytes.length; ++id)
      {
        result[id] -= mean;
      }
    }

    return result;
  }

  /**
   * forward interpretation, from feature to score for each class(which is not returned)
   *
   * @param feature
   */
  public void forward(byte[] feature)
  {
    if(this._interpretations == null)
    {
      this._interpretations = new Matrix[this._num_hidden_layer + 1];
    }
    if(this._activations == null)
    {
      this._activations = new Matrix[this._num_hidden_layer + 1];
    }
    this._activations[0] = new Matrix(this._num_input, 1, this.normalize(feature));
    for(int idl = 0; idl <= this._num_hidden_layer; ++idl)
    {
      this._interpretations[idl] = this._weights[idl].multiply(
          this._activations[idl].appendAsVecBias());
      if(idl < this._num_hidden_layer)
      {
        this._activations[idl + 1] = this._interpretations[idl].relu();
      }
    }
    this._interpretations[this._num_hidden_layer].softmaxAsVecOnSelf();
  }

  /**
   * back propagation
   *
   * @param result
   */
  public void backward(int result)
  {
    Matrix diff = new Matrix(this._num_output, 1,
        this._interpretations[this._num_hidden_layer].getData());
    diff.addElement(result, 0, -1.0f);
    for(int idl = this._num_hidden_layer; idl >= 0; --idl)
    {
      if(idl < this._num_hidden_layer)
      {
        diff.multiplyElemWiseOnSelf(this._interpretations[idl].reluDer());
      }
      Matrix delta = diff.ger(this._activations[idl].appendAsVecBias());
      if(idl > 0)
      {
        diff.conjAsVecOnSelf();
        diff = this._weights[idl].multiplyLeft(diff);
        diff = new Matrix(diff.getNumCol() - 1, 1,
            Arrays.copyOf(diff.getData(), diff.getNumCol() - 1));
      }
      this._weights[idl].addOnSelf(delta, -this._stepsize);
    }

    ++this._count_back_prop;
    if(this._count_back_prop >= this._num_iter_to_damp_stepsize)
    {
      this._stepsize *= 0.5f;
      this._count_back_prop = 0;
    }
  }

  public static int getResult(float[] likelihood, float min_score)
  {
    float max = min_score;
    int id_max = -1;
    for(int i = 0; i < likelihood.length; ++i)
    {
      if(likelihood[i] > max)
      {
        id_max = i;
        max = likelihood[i];
      }
    }
    return id_max;
  }

  /**
   * return the result of classification, index of highest score
   *
   * @return
   */
  public int getResult()
  {
    return this.getResult(0.0f);
  }

  public int getResult(float min_score)
  {
    return Network.getResult(this._interpretations[this._num_hidden_layer].getData(), min_score);
  }

  /**
   * return fake entropy, sum of square error between score and ideal score(01000 etc)
   *
   * @return
   */
  public float getEntropy()
  {
    float entropy = 0.0f;
    final float[] eval = this._interpretations[this._num_hidden_layer].getData();
    final int length = this._num_output;
    for(int i = 0; i < length; ++i)
    {
      entropy += (float)Math.log((double) eval[i]);
    }
    return -entropy / ((float)length * (float)Math.log(2.0));
  }

  /**
   * @return
   */
  public int getNumLayer()
  {
    return this._weights.length;
  }

  /**
   * return weights
   *
   * @param id
   * @return
   */
  public Matrix getLayer(int id)
  {
    assert (id < this._weights.length);
    return this._weights[id];
  }

  /**
   * set the size of input and output of network
   *
   * @param num_in
   * @param num_out
   */
  public void setDimensionality(int num_in, int num_out)
  {
    this._num_input = num_in;
    this._num_output = num_out;
  }

  /**
   * set the weights and other members for forwards from weights(mats)
   *
   * @param mats
   */
  public void setLayers(Matrix[] mats)
  {
    assert (mats.length > 0);
    for(int i = 1; i < mats.length; ++i)
    {
      assert (mats[i - 1].getNumRow() == mats[i].getNumCol() + 1);
    }

    this._num_input = mats[0].getNumCol() - 1;
    this._num_output = mats[mats.length - 1].getNumRow();
    this._num_hidden_layer = mats.length - 1;
    this._weights = new Matrix[mats.length];
    for(int i = 0; i < mats.length; ++i)
    {
      this._weights[i] = new Matrix(mats[i]);
    }
  }

  public boolean isNetworkNormal()
  {
    boolean is_okay = true;
    for(Matrix w : this._weights)
    {
      if(!w.isNormal())
      {
        is_okay = false;
        break;
      }
    }
    return is_okay;
  }

  public int[] getDemensionality()
  {
    return new int[]{this._num_input, this._num_output};
  }

  public float[] getLikelihood()
  {
    return this._interpretations[this._num_hidden_layer].getData();
  }
}
