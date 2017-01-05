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
    if(is_random)
    {
      Random rand = new Random();
      float range = (float) (Math.sqrt(6.0) / (float) Math.sqrt((double) (num_col + num_row)));
      for(int i = 0; i < this.data.length; ++i)
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
    if(is_add_bias)
    {
      this.num_row = num_row + 1;
    }
    else
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
  public void multiplyElemWiseOnSelf(Matrix another)
  {
    assert (this.num_row == another.num_row);
    assert (this.num_col == another.num_col);
    final int len = this.data.length;
    for(int i = 0; i < len; ++i)
    {
      this.data[i] *= another.data[i];
    }
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
    for(int r = 0; r < this.num_row; ++r)
    {
      int row_start = r * vec_b.num_row;
      for(int c = 0; c < vec_b.num_row; ++c)
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
    final int len = this.data.length;
    for(int i = 0; i < len; ++i)
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
  public Matrix multiply(Matrix another)
  {
    assert (this.num_col == another.num_row);
    final int num_row = this.num_row;
    final int num_col = another.getNumCol();
    final int num_dua = another.getNumRow();
    Matrix result = new Matrix(num_row, num_col);
    if(num_col == 1)
    {
      // matrix by vector
      for(int r = 0; r < num_row; ++r)
      {
        final int row_start = r * num_dua;
        float val = 0.0f;
        for(int i = 0; i < num_dua; ++i)
        {
          val += this.data[row_start + i] * another.data[i];
        }
        result.data[r] = val;
      }
    }
    else if(num_row == 1)
    {
      // vector by matrix
      int[] col_start = this.genColStart();
      for(int c = 0; c < num_col; ++c)
      {
        float val = 0.0f;
        for(int i = 0; i < num_dua; ++i)
        {
          val += this.data[i] * another.data[col_start[i] + c];
        }
        result.data[c] = val;
      }
    }
    else
    {
      // matrix by matrix
      int[] col_start = this.genColStart();
      for(int r = 0; r < num_row; ++r)
      {
        final int row_start = r * num_dua;
        for(int c = 0; c < num_col; ++c)
        {
          float val = 0.0f;
          for(int i = 0; i < num_dua; ++i)
          {
            val += this.data[row_start + i] * another.data[col_start[i] + c];
          }
          result.data[col_start[r] + c] = val;
        }
      }
    }
    return result;
  }

  /**
   * return another * this, use this when left matrix is smaller
   *
   * @param another
   * @return
   */
  public Matrix multiplyLeft(Matrix another)
  {
    assert (another.num_col == another.num_row);
    final int num_row = another.num_row;
    final int num_col = this.getNumCol();
    final int num_dua = this.getNumRow();
    Matrix result = new Matrix(num_row, num_col);
    for(int r = 0, row_start = 0; r < num_row; ++r, row_start += num_dua)
    {
      for(int c = 0; c < num_col; ++c)
      {
        float val = 0.0f;
        for(int i = 0, col_start = c; i < num_dua; ++i, col_start += num_col)
        {
          val += another.data[i] * this.data[col_start];
        }
        result.data[row_start + c] = val;
      }
    }
    return result;
  }

  /**
   * return relu of this
   *
   * @return
   */
  public Matrix relu()
  {
    Matrix result = new Matrix(this.num_row, this.num_col, this.data);
    final int len = result.data.length;
    for(int i = 0; i < len; ++i)
    {
      if(result.data[i] < 0.0f)
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
  public Matrix reluDer()
  {
    Matrix result = new Matrix(this.num_row, this.num_col, this.data);
    final int len = result.data.length;
    for(int i = 0; i < len; ++i)
    {
      if(result.data[i] >= 0.0f)
      {
        result.data[i] = 1.0f;
      }
      else
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
    final int len = this.data.length;
    for(int i = 0; i < len; ++i)
    {
      this.data[i] = (float) Math.exp((double) this.data[i]);
      sum += this.data[i];
    }
    for(int i = 0; i < len; ++i)
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
    for(int i = 0; i < len; ++i)
    {
      result[i] = this_;
      this_ += diff;
    }
    return result;
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
  public Matrix conv(Matrix kernel, boolean is_smoothing)
  {
    Matrix result = new Matrix(this.num_row, this.num_col);
    final int krr = kernel.num_row / 2;
    final int krc = kernel.num_col / 2;

    final int num_row = this.num_row;
    final int num_col = this.num_col;
    final int num_row_kernel = kernel.num_row;
    final int num_col_kernel = kernel.num_col;
    int idx = 0;
    for(int ir = 0; ir < num_row; ++ir)
    {
      final int r = ir - krr;
      for(int ic = 0; ic < num_col; ++ic)
      {
        final int c = ic - krc;
        float val = 0.0f;
        float weight = 0.0f;
        int idx_k = 0;
        for(int ikr = 0; ikr < num_row_kernel; ++ikr)
        {
          final int rr = r + ikr;
          final int row_start = rr * num_col;
          for(int ikc = 0; ikc < num_col_kernel; ++ikc)
          {
            int rc = c + ikc;
            if(rr >= 0 && rc >= 0 && rr < num_row && rc < num_col)
            {
              val += kernel.data[idx_k] * this.data[row_start + rc];
              weight += kernel.data[idx_k];
            }
            ++idx_k;
          }
        }
        if(is_smoothing)
        {
          val /= weight;
        }
        result.data[idx] = val;
        ++idx;
      }
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

    final int len = this.data.length;
    for(int i = 0; i < len; ++i)
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
    for(int r = 0; r < b; ++r)
    {
      final int start = r * this.num_col;
      for(int c = 0; c < this.num_col; ++c)
      {
        this.data[start + c] = val;
      }
    }
    // bottom
    for(int r = this.num_row - 1; r >= this.num_row - 1 - b; --r)
    {
      final int start = r * this.num_col;
      for(int c = 0; c < this.num_col; ++c)
      {
        this.data[start + c] = val;
      }
    }
    // left
    for(int c = 0; c < b; ++c)
    {
      int start = c;
      for(int r = 0; r < this.num_row; ++r)
      {
        this.data[start] = val;
        start += this.num_col;
      }
    }
    // right
    for(int c = this.num_col - 1; c >= this.num_col - 1 - b; --c)
    {
      int start = c;
      for(int r = 0; r < this.num_row; ++r)
      {
        this.data[start] = val;
        start += this.num_col;
      }
    }
  }

  /**
   * set values to [0, 1]
   */
  public void normalizeOnSelf()
  {
    float min = this.data[0];
    float max = this.data[0];
    final int len = this.data.length;
    float scale;
    for(float f : this.data)
    {
      if(min > f)
      {
        min = f;
      }
      if(max < f)
      {
        max = f;
      }
    }
    scale = 1.0f / (max - min);
    for(int i = 0; i < len; ++i)
    {
      this.data[i] = (this.data[i] - min) * scale;
    }
  }

  private int[] getSortOrder(List<Float> list)
  {
    List<Integer> order = new LinkedList<Integer>();
    List<Float> sorted = new LinkedList<Float>();

    int index = 0;
    for(Float f : list)
    {
      if(order.isEmpty())
      {
        sorted.add(f);
        order.add(index);
      }
      else
      {
        int l = 0;
        int r = sorted.size();
        int i;
        float fval = f.floatValue();
        while(r - l > 16)
        {
          i = (l + r) / 2;
          if(sorted.get(i).floatValue() > fval)
          {
            l = i - 1;
          }
          else
          {
            r = i;
          }
        }
        for(i = l; i < r && sorted.get(i).floatValue() > fval; ++i)
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
    for(Integer i : order)
    {
      result[id_order] = i.intValue();
      ++id_order;
    }
    return result;
  }

  public int[][] getLocalMaxima(int radius, float minimal, boolean is_sorted)
  {
    List<Float> maximas = new ArrayList<Float>();
    List<Integer> r_max = new ArrayList<Integer>();
    List<Integer> c_max = new ArrayList<Integer>();

    final int len_r_loop = this.num_row;
    final int len_c_loop = this.num_col;
    for(int r = 0; r < len_r_loop; ++r)
    {
      final int row_start_out = r * this.num_col;
      for(int c = 0; c < len_c_loop; ++c)
      {
        final float val = this.data[row_start_out + c];
        if(val < minimal)
        {
          continue;
        }
        boolean is_max = true;
        for(int lr = -radius; lr <= radius; ++lr)
        {
          final int this_r = r + lr;
          if(this_r >= 0 && this_r < len_r_loop)
          {
            final int row_start = this_r * this.num_col;
            for(int lc = -radius; lc <= radius; ++lc)
            {
              final int this_c = c + lc;
              if(this_c >= 0 && this_c < len_c_loop && val < this.data[row_start + this_c])
              {
                is_max = false;
                break;
              }
            }
            if(!is_max)
            {
              break;
            }
          }
        }

        if(is_max)
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
    for(int i : sorted_order)
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
    for(int id = 0; id < index.length; ++id)
    {
      assert (index[id].length == 2);
      result[id] = this.data[index[id][0] * this.num_col + index[id][1]];
    }
    return result;
  }

  public void thresholdOnSelf(float thres)
  {
    final int len = this.data.length;
    for(int i = 0; i < len; ++i)
    {
      if(this.data[i] < thres)
      {
        this.data[i] = 0.0f;
      }
      else
      {
        this.data[i] = 1.0f;
      }
    }
  }

  public void erodeOnSelf(int radius, float thres)
  {
    final int nr_1 = this.num_row - radius;
    final int nc_1 = this.num_col - radius;
    float[] copy = Arrays.copyOf(this.data, this.data.length);
    Arrays.fill(this.data, 0.0f);
    for(int r = radius; r < nr_1; ++r)
    {
      final int r_s = r * this.num_col;
      for(int c = radius; c < nc_1; ++c)
      {
        boolean is_all_above = true;
        for(int rr = -radius; rr <= radius && is_all_above; ++rr)
        {
          int rs = (rr + r) * this.num_col + c;
          for(int cc = -radius; cc <= radius && is_all_above; ++cc)
          {
            if(copy[rs + cc] < thres)
            {
              is_all_above = false;
            }
          }
        }
        if(is_all_above)
        {
          this.data[c + r_s] = copy[c + r_s];
        }
      }
    }
  }
}
