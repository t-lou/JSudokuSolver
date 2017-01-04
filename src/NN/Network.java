package NN;

import java.util.Arrays;

public class Network
{
  private int num_hidden_layer;
  private int num_input;
  private int num_output;
  private float stepsize;
  private float stepsize_base;
  private int count_back_prop;
  private int num_iter_to_damp_stepsize;
  private Matrix[] weights;
  private Matrix[] interpretations;
  private Matrix[] activations;

  public Network()
  {
  }

  public Network(int num_hidden_layer, int num_hidden_unit,
                 int num_input, int num_output, float stepsize)
  {
    this.num_hidden_layer = num_hidden_layer;
    this.num_input = num_input;
    this.num_output = num_output;
    this.weights = new Matrix[num_hidden_layer + 1];
    this.stepsize_base = stepsize;
    this.stepsize = this.stepsize_base;
    this.count_back_prop = 0;

    // last row is for bias
    this.weights[0] = new Matrix(num_hidden_unit, num_input + 1, true);
    for(int i = 1; i < num_hidden_layer; ++i)
    {
      this.weights[i] = new Matrix(num_hidden_unit, num_hidden_unit + 1, true);
    }
    this.weights[num_hidden_layer] = new Matrix(num_output, num_hidden_unit + 1, true);
    this.num_iter_to_damp_stepsize = 0;
    for(Matrix w : this.weights)
    {
      this.num_iter_to_damp_stepsize += w.getNumCol() * w.getNumRow();
    }
    this.num_iter_to_damp_stepsize *= 2;
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

    for(int id = 0; id < bytes.length; ++id)
    {
      result[id] = (result[id] - mean) / dev;
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
    if(this.interpretations == null)
    {
      this.interpretations = new Matrix[this.num_hidden_layer + 1];
    }
    if(this.activations == null)
    {
      this.activations = new Matrix[this.num_hidden_layer + 1];
    }
    this.activations[0] = new Matrix(this.num_input, 1, this.normalize(feature));
    for(int idl = 0; idl <= this.num_hidden_layer; ++idl)
    {
      this.interpretations[idl] = this.weights[idl].multiply(
          this.activations[idl].appendAsVecBias());
      if(idl < this.num_hidden_layer)
      {
        this.activations[idl + 1] = this.interpretations[idl].relu();
      }
    }
    this.interpretations[this.num_hidden_layer].softmaxAsVecOnSelf();
  }

  /**
   * back propagation
   *
   * @param result
   */
  public void backward(int result)
  {
    Matrix diff = new Matrix(this.num_output, 1,
        this.interpretations[this.num_hidden_layer].getData());
    diff.addElement(result, 0, -1.0f);
    for(int idl = this.num_hidden_layer; idl >= 0; --idl)
    {
      if(idl < this.num_hidden_layer)
      {
        diff.multiplyElemWiseOnSelf(this.interpretations[idl].reluDer());
      }
      Matrix delta = diff.ger(this.activations[idl].appendAsVecBias());
      if(idl > 0)
      {
        diff.conjAsVecOnSelf();
        diff = this.weights[idl].multiplyLeft(diff);
        diff = new Matrix(diff.getNumCol() - 1, 1,
            Arrays.copyOf(diff.getData(), diff.getNumCol() - 1));
      }
      this.weights[idl].addOnSelf(delta, -this.stepsize);
    }

    ++this.count_back_prop;
    if(this.count_back_prop >= this.num_iter_to_damp_stepsize)
    {
      this.stepsize *= 0.5f;
      this.count_back_prop = 0;
    }
  }

  /**
   * return the result of classification, index of highest score
   *
   * @return
   */
  public int getResult()
  {
    float max = -1.0f;
    int id_max = -1;
    float[] eval = this.interpretations[this.num_hidden_layer].getData();
    for(int i = 0; i < this.num_output; ++i)
    {
      if(max < eval[i])
      {
        id_max = i;
        max = eval[i];
      }
    }
    return id_max;
  }

  /**
   * return fake entropy, sum of square error between score and ideal score(01000 etc)
   *
   * @return
   */
  public float getFakeEntropy()
  {
    float err = 0.0f;
    float[] eval = this.interpretations[this.num_hidden_layer].getData();
    for(int i = 0; i < this.num_output; ++i)
    {
      if(eval[i] < 0.5f)
      {
        err += eval[i] * eval[i];
      }
      else
      {
        err += (1.0f - eval[i]) * (1.0f - eval[i]);
      }
    }
    return err;
  }

  /**
   * @return
   */
  public int getNumLayer()
  {
    return this.weights.length;
  }

  /**
   * return weights
   *
   * @param id
   * @return
   */
  public Matrix getLayer(int id)
  {
    assert (id < this.weights.length);
    return this.weights[id];
  }

  /**
   * set the size of input and output of network
   *
   * @param num_in
   * @param num_out
   */
  public void setDimensionality(int num_in, int num_out)
  {
    this.num_input = num_in;
    this.num_output = num_out;
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

    this.num_input = mats[0].getNumCol() - 1;
    this.num_output = mats[mats.length - 1].getNumRow();
    this.num_hidden_layer = mats.length - 1;
    this.weights = new Matrix[mats.length];
    for(int i = 0; i < mats.length; ++i)
    {
      this.weights[i] = new Matrix(mats[i]);
    }
  }
}
