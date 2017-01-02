package NN;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;

public class Matrix
{
  private int num_row, num_col;
  private float[] data;

  public Matrix()
  {
    this(0, 0, false);
  }

  public Matrix(Matrix mat)
  {
    this(mat.num_row, mat.num_col, mat.data, false);
  }

  public Matrix(int num_row, int num_col)
  {
    this(num_row, num_col, false);
  }

  public Matrix(int num_row, int num_col, boolean is_random)
  {
    this.data = new float[num_row * num_col];
    this.num_col = num_col;
    this.num_row = num_row;
    if (is_random)
    {
      Random rand = new Random();
      float range = (float) (Math.sqrt(6.0)
          / (float) Math.sqrt((double) (num_col + num_row)));
      for (int i = 0; i < this.data.length; ++i)
      {
        this.data[i] = (rand.nextFloat() - 0.5f) * range;
      }
    }
  }

  public Matrix(int num_row, int num_col, float[] values)
  {
    this(num_row, num_col, values, false);
  }

  public Matrix(int num_row, int num_col, float[] values, boolean is_add_bias)
  {
    if (is_add_bias)
    {
      this.num_row = num_row + 1;
    } else
    {
      this.num_row = num_row;
    }
    this.num_col = num_col;
    this.data = new float[this.num_col * this.num_row];
    Arrays.fill(this.data, 1.0f);
    System.arraycopy(values, 0, this.data, 0, num_row * num_col);
  }

  /**
   * multiply another element-wise on this
   *
   * @param another
   */
  public void multElemWiseOnSelf(Matrix another)
  {
    assert (this.num_row == another.num_row);
    assert (this.num_col == another.num_col);
    int len = this.data.length;
    for (int i = 0; i < len; ++i)
    {
      this.data[i] *= another.data[i];
    }
  }

  private int[] genBlockPlan(int len, int blk)
  {
    int num_blk = len / blk;
    int[] plan;
    if (num_blk > 0)
    {
      int rest = len - num_blk * blk;
      plan = new int[num_blk];
      Arrays.fill(plan, blk);
      plan[num_blk - 1] += rest;
    } else
    {
      plan = new int[1];
      plan[0] = len;
    }
    return plan;
  }

  /**
   * suppose this is r-length vector, vec_b is c-length vector,
   * return (r,c) matrix of this*vec_b.T
   *
   * @param vec_b
   * @return
   */
  public Matrix ger(Matrix vec_b)
  {
    assert (this.num_col == 1);
    assert (vec_b.num_col == 1);
    Matrix result = new Matrix(this.num_row, vec_b.num_row);
    for (int r = 0; r < this.num_row; ++r)
    {
      int row_start = r * vec_b.num_row;
      for (int c = 0; c < vec_b.num_row; ++c)
      {
        result.data[row_start + c] = this.data[r] * vec_b.data[c];
      }
    }
    return result;
  }

  /**
   * add delta on this, this := this + scale * delta
   *
   * @param delta
   * @param scale
   */
  public void addOnSelf(Matrix delta, float scale)
  {
    assert (this.num_col == delta.num_col);
    assert (this.num_row == delta.num_row);
    int len = this.data.length;
    for (int i = 0; i < len; ++i)
    {
      this.data[i] += delta.data[i] * scale;
    }
  }

  /**
   * return this * another
   *
   * @param another
   * @return
   */
  public Matrix mult(Matrix another)
  {
    int num_row = this.num_row;
    int num_col = another.getNumCol();
    int num_dua = another.getNumRow();
    assert (this.num_col == num_dua);

    float[] values = new float[num_row * num_col];
    Arrays.fill(values, 0.0f);
    // naive matrix multiplication
    if (num_col == 1)
    {
      // matrix by vector
      for (int r = 0; r < num_row; ++r)
      {
        int row_start = r * num_dua;
        float val = 0.0f;
        for (int i = 0; i < num_dua; ++i)
        {
          val += this.data[row_start + i] * another.data[i];
        }
        values[r] = val;
      }
    } else if (num_row == 1)
    {
      // vector by matrix
      int[] col_start = this.genColStart();
      for (int c = 0; c < num_col; ++c)
      {
        float val = 0.0f;
        for (int i = 0; i < num_dua; ++i)
        {
          val += this.data[i] * another.data[col_start[i] + c];
        }
        values[c] = val;
      }
    } else
    {
      // matrix by matrix
      int[] col_start = this.genColStart();
      for (int r = 0; r < num_row; ++r)
      {
        int row_start = r * num_dua;
        for (int c = 0; c < num_col; ++c)
        {
          float val = 0.0f;
          for (int i = 0; i < num_dua; ++i)
          {
            val += this.data[row_start + i] *
                another.data[col_start[i] + c];
          }
          values[col_start[r] + c] = val;
        }
      }
    }
    return new Matrix(num_row, num_col, values);
  }

  /**
   * return relu of this
   *
   * @return
   */
  public Matrix relu()
  {
    Matrix result = new Matrix(this.num_row, this.num_col, this.data);
    int len = result.data.length;
    for (int i = 0; i < len; ++i)
    {
      if (result.data[i] < 0.0f)
      {
        result.data[i] = 0.0f;
      }
    }
    return result;
  }

  /**
   * return derivative of relu of this
   *
   * @return
   */
  public Matrix relu_der()
  {
    Matrix result = new Matrix(this.num_row, this.num_col, this.data);
    int len = result.data.length;
    for (int i = 0; i < len; ++i)
    {
      if (result.data[i] >= 0.0f)
      {
        result.data[i] = 1.0f;
      } else
      {
        result.data[i] = 0.0f;
      }
    }
    return result;
  }

  /**
   * this := softmax(this), suppose this is vector (1 column)
   */
  public void softmaxAsVecOnSelf()
  {
    assert (this.num_row == 1 || this.num_col == 1);
    float sum = 0.0f;
    int len = this.data.length;
    for (int i = 0; i < len; ++i)
    {
      this.data[i] = (float) Math.exp((double) this.data[i]);
      sum += this.data[i];
    }
    for (int i = 0; i < len; ++i)
    {
      this.data[i] /= sum;
    }
  }

  /**
   * this := this.T, suppose this is vector (1 column)
   */
  public void conjAsVecOnSelf()
  {
    assert (this.num_row == 1 || this.num_col == 1);
    int num = this.num_row;
    this.num_row = this.num_col;
    this.num_col = num;
  }

  /**
   * this(row, col) := this(row, col) + delta
   *
   * @param row
   * @param col
   * @param delta
   */
  public void addElement(int row, int col, float delta)
  {
    assert (row < this.num_row);
    assert (col < this.num_col);
    this.data[row * this.num_col + col] += delta;
  }

  /**
   * this := (this, 1), suppose this is vector (1 column)
   *
   * @return
   */
  public Matrix appendAsVecBias()
  {
    assert (this.num_col == 1);
    return new Matrix(this.num_row, this.num_col, this.data, true);
  }

  /**
   * return data(float[])
   *
   * @return
   */
  public float[] getData()
  {
    return this.data;
  }

  /**
   * return number of columns
   *
   * @return
   */
  public int getNumCol()
  {
    return this.num_col;
  }

  /**
   * return number of rows
   *
   * @return
   */
  public int getNumRow()
  {
    return this.num_row;
  }

  private int[] genArithmeticSequence(int len, int diff)
  {
    int[] result = new int[len];
    int this_ = 0;
    for (int i = 0; i < len; ++i)
    {
      result[i] = this_;
      this_ += diff;
    }
    return result;
  }

  private int[] genRowStart()
  {
    return this.genArithmeticSequence(this.num_row, this.num_col);
  }

  private int[] genColStart()
  {
    return this.genArithmeticSequence(this.num_col, this.num_row);
  }

  /**
   * 2d convolution
   *
   * @param kernel
   * @return
   */
  public Matrix conv(Matrix kernel)
  {
    int blk = 32;
    int[] plan_r = this.genBlockPlan(this.num_row, blk);
    int[] plan_c = this.genBlockPlan(this.num_col, blk);
    Matrix result = new Matrix(this.num_row, this.num_col);
//		Arrays.fill(result.data, 0.0f);
    int krr = kernel.num_row / 2;
    int krc = kernel.num_col / 2;

    int r_offset = 0;
    for (int rb : plan_r)
    {
      int c_offset = 0;
      for (int cb : plan_c)
      {
        for (int ir = 0; ir < rb; ++ir)
        {
          int r = r_offset + ir - krr;
          for (int ic = 0; ic < cb; ++ic)
          {
            int c = c_offset + ic - krc;
            float val = 0.0f;
            int idx_k = 0;
            for (int ikr = 0; ikr < kernel.num_row; ++ikr)
            {
              for (int ikc = 0; ikc < kernel.num_col; ++ikc)
              {
                int rr = r + ikr;
                int rc = c + ikc;
                if (rr >= 0 && rc >= 0 && rr < this.num_row && rc < this.num_col)
                {
                  val += kernel.data[idx_k] *
                      this.data[rr * this.num_col + rc];
                }
                ++idx_k;
              }
            }
            result.data[(r_offset + ir) * this.num_col + c_offset + ic] = val;
          }
        }
        c_offset += cb;
      }
      r_offset += rb;
    }

    return result;
  }

  /**
   * this := |this| + |mat|
   *
   * @param mat
   */
  public void addAbsElemWiseOnSelf(Matrix mat)
  {
    assert (mat.num_col == this.num_col);
    assert (mat.num_row == this.num_row);

    for (int i = 0; i < this.data.length; ++i)
    {
      this.data[i] = Math.abs(this.data[i]) + Math.abs(mat.data[i]);
    }
  }

  /**
   * set boundary to val
   *
   * @param val
   */
  public void setBoundary(float val)
  {
    int b = 10;
    // upper
    for (int r = 0; r < b; ++r)
    {
      int start = r * this.num_col;
      for (int c = 0; c < this.num_col; ++c)
      {
        this.data[start + c] = val;
      }
    }
    // bottom
    for (int r = this.num_row - 1; r >= this.num_row - 1 - b; --r)
    {
      int start = r * this.num_col;
      for (int c = 0; c < this.num_col; ++c)
      {
        this.data[start + c] = val;
      }
    }
    // left
    for (int c = 0; c < b; ++c)
    {
      int start = c;
      for (int r = 0; r < this.num_row; ++r)
      {
        this.data[start] = val;
        start += this.num_col;
      }
    }
    // right
    for (int c = this.num_col - 1; c >= this.num_col - 1 - b; --c)
    {
      int start = c;
      for (int r = 0; r < this.num_row; ++r)
      {
        this.data[start] = val;
        start += this.num_col;
      }
    }
  }

  public void rotateClockwise()
  {
    float[] tmp = new float[this.data.length];
    System.arraycopy(this.data, 0, tmp, 0, tmp.length);

    for (int r = 0; r < this.num_row; ++r)
    {
      int c1 = this.num_row - 1 - r;
      for (int c = 0; c < this.num_col; ++c)
      {
        this.data[(this.num_col - 1 - c) * this.num_row + c1] = tmp[r * this.num_col + c];
      }
    }

    int t = this.num_col;
    this.num_col = this.num_row;
    this.num_row = t;
  }

  /**
   * set values to [0, 1]
   */
  public void normalizeOnSelf()
  {
    float min = this.data[0];
    float max = this.data[0];
    int len = this.data.length;
    float scale;
    for (float f : this.data)
    {
      if (min > f)
      {
        min = f;
      }
      if (max < f)
      {
        max = f;
      }
    }
    scale = 1.0f / (max - min);
    for (int i = 0; i < len; ++i)
    {
      this.data[i] = (this.data[i] - min) * scale;
    }
  }

  private int[] getSortOrder(List<Float> list)
  {
    List<Integer> order = new LinkedList<Integer>();
    List<Float> sorted = new LinkedList<Float>();

    int index = 0;
    for (Float f : list)
    {
      if (order.isEmpty())
      {
        sorted.add(f);
        order.add(index);
      } else
      {
        int l = 0;
        int r = sorted.size();
        int i;
        float fval = f.floatValue();
        while (r - l > 16)
        {
          i = (l + r) / 2;
          if (sorted.get(i).floatValue() > fval)
          {
            l = i - 1;
          } else
          {
            r = i;
          }
        }
        for (i = l; i < r && sorted.get(i).floatValue() > fval; ++i)
        {
        }
        sorted.add(i, f);
        order.add(i, index);
      }
      ++index;
    }

    int size_order = order.size();
    int[] result = new int[size_order];
    int id_order = 0;
    for (Integer i : order)
    {
      result[id_order] = i.intValue();
      ++id_order;
    }
    return result;
  }

  public int[][] getLocalMaxima()
  {
    List<Float> maximas = new ArrayList<Float>();
    List<Integer> r_max = new ArrayList<Integer>();
    List<Integer> c_max = new ArrayList<Integer>();

    int radius = 4;
    int len_r_loop = this.num_row - radius;
    int len_c_loop = this.num_col - radius;
    for (int r = radius; r < len_r_loop; ++r)
    {
      int row_start_out = r * this.num_col;
      for (int c = radius; c < len_c_loop; ++c)
      {
        float val = this.data[row_start_out + c];
        if (val < 0.5f)
        {
          continue;
        }
        boolean is_max = true;
        for (int lr = -radius; lr <= radius; ++lr)
        {
          int row_start = (r + lr) * this.num_col;
          for (int lc = -radius; lc <= radius; ++lc)
          {
            if (val < this.data[row_start + (c + lc)])
            {
              is_max = false;
              break;
            }
          }
          if (!is_max)
          {
            break;
          }
        }

        if (is_max)
        {
          maximas.add(val);
          r_max.add(r);
          c_max.add(c);
        }
      }
    }

    int[][] result = new int[maximas.size()][2];
    int[] sorted_order = this.getSortOrder(maximas);
    int index = 0;
    for (int i : sorted_order)
    {
      result[index][0] = r_max.get(i).intValue();
      result[index][1] = c_max.get(i).intValue();
      ++index;
    }
    return result;
  }

  public float[] getElement(int[][] index)
  {
    float[] result = new float[index.length];
    for (int id = 0; id < index.length; ++id)
    {
      assert (index[id].length == 2);
      result[id] = this.data[index[id][0] * this.num_col + index[id][1]];
    }
    return result;
  }
}
