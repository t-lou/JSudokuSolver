package NN;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;

public class Matrix
{
  private int _num_row, _num_col;
  private float[] _data;

  public Matrix()
  {
    this(0, 0, false);
  }

  public Matrix(Matrix mat)
  {
    this(mat._num_row, mat._num_col, mat._data, false);
  }

  public Matrix(int num_row, int num_col)
  {
    this(num_row, num_col, false);
  }

  public Matrix(int num_row, int num_col, boolean is_random)
  {
    this._data = new float[num_row * num_col];
    this._num_col = num_col;
    this._num_row = num_row;
    if(is_random)
    {
      Random rand = new Random();
      float range = (float) (Math.sqrt(6.0) / (float) Math.sqrt((double) (num_col + num_row)));
      for(int i = 0; i < this._data.length; ++i)
      {
        this._data[i] = (rand.nextFloat() - 0.5f) * range;
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
      this._num_row = num_row + 1;
    }
    else
    {
      this._num_row = num_row;
    }
    this._num_col = num_col;
    this._data = new float[this._num_col * this._num_row];
    Arrays.fill(this._data, 1.0f);
    System.arraycopy(values, 0, this._data, 0, num_row * num_col);
  }

  /**
   * multiply another element-wise on this
   *
   * @param another
   */
  public void multiplyElemWiseOnSelf(Matrix another)
  {
    assert (this._num_row == another._num_row);
    assert (this._num_col == another._num_col);
    final int len = this._data.length;
    for(int i = 0; i < len; ++i)
    {
      this._data[i] *= another._data[i];
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
    assert (this._num_col == 1);
    assert (vec_b._num_col == 1);
    Matrix result = new Matrix(this._num_row, vec_b._num_row);
    for(int r = 0; r < this._num_row; ++r)
    {
      int row_start = r * vec_b._num_row;
      for(int c = 0; c < vec_b._num_row; ++c)
      {
        result._data[row_start + c] = this._data[r] * vec_b._data[c];
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
    assert (this._num_col == delta._num_col);
    assert (this._num_row == delta._num_row);
    final int len = this._data.length;
    for(int i = 0; i < len; ++i)
    {
      this._data[i] += delta._data[i] * scale;
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
    assert (this._num_col == another._num_row);
    final int num_row = this._num_row;
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
          val += this._data[row_start + i] * another._data[i];
        }
        result._data[r] = val;
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
          val += this._data[i] * another._data[col_start[i] + c];
        }
        result._data[c] = val;
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
            val += this._data[row_start + i] * another._data[col_start[i] + c];
          }
          result._data[col_start[r] + c] = val;
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
    assert (another._num_col == another._num_row);
    final int num_row = another._num_row;
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
          val += another._data[i] * this._data[col_start];
        }
        result._data[row_start + c] = val;
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
    Matrix result = new Matrix(this._num_row, this._num_col, this._data);
    final int len = result._data.length;
    for(int i = 0; i < len; ++i)
    {
      if(result._data[i] < 0.0f)
      {
        result._data[i] = 0.0f;
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
    Matrix result = new Matrix(this._num_row, this._num_col, this._data);
    final int len = result._data.length;
    for(int i = 0; i < len; ++i)
    {
      if(result._data[i] >= 0.0f)
      {
        result._data[i] = 1.0f;
      }
      else
      {
        result._data[i] = 0.0f;
      }
    }
    return result;
  }

  /**
   * this := softmax(this), suppose this is vector (1 column)
   */
  public void softmaxAsVecOnSelf()
  {
    assert (this._num_row == 1 || this._num_col == 1);
    float sum = 0.0f;
    final int len = this._data.length;
    for(int i = 0; i < len; ++i)
    {
      this._data[i] = (float) Math.exp((double) this._data[i]);
      sum += this._data[i];
    }
    for(int i = 0; i < len; ++i)
    {
      this._data[i] /= sum;
    }
  }

  public Matrix conj()
  {
    Matrix mat = new Matrix(this._num_col, this._num_row);
    final int[] css = this.genColStart();
    for(int r = 0, rs = 0; r < this._num_row; ++r, rs += this._num_col)
    {
      for(int c = 0; c < this._num_col; ++c)
      {
        mat._data[css[c] + r] = this._data[rs + c];
      }
    }
    return mat;
  }

  /**
   * this := this.T, suppose this is vector (1 column)
   */
  public void conjAsVecOnSelf()
  {
    assert (this._num_row == 1 || this._num_col == 1);
    int num = this._num_row;
    this._num_row = this._num_col;
    this._num_col = num;
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
    assert (row < this._num_row);
    assert (col < this._num_col);
    this._data[row * this._num_col + col] += delta;
  }

  /**
   * this := (this, 1), suppose this is vector (1 column)
   *
   * @return
   */
  public Matrix appendAsVecBias()
  {
    assert (this._num_col == 1);
    return new Matrix(this._num_row, this._num_col, this._data, true);
  }

  /**
   * return data(float[])
   *
   * @return
   */
  public float[] getData()
  {
    return this._data;
  }

  /**
   * return number of columns
   *
   * @return
   */
  public int getNumCol()
  {
    return this._num_col;
  }

  /**
   * return number of rows
   *
   * @return
   */
  public int getNumRow()
  {
    return this._num_row;
  }

  private int[] genArithmeticSequence(int len, int diff)
  {
    int[] result = new int[len];
    for(int i = 0, this_ = 0; i < len; ++i, this_ += diff)
    {
      result[i] = this_;
    }
    return result;
  }

  private int[] genColStart()
  {
    return this.genArithmeticSequence(this._num_col, this._num_row);
  }

  /**
   * 2d convolution
   *
   * @param kernel
   * @return
   */
  public Matrix conv(Matrix kernel, boolean is_smoothing)
  {
    Matrix result = new Matrix(this._num_row, this._num_col);
    final int krr = kernel._num_row / 2;
    final int krc = kernel._num_col / 2;

    final int num_row = this._num_row;
    final int num_col = this._num_col;
    final int num_row_kernel = kernel._num_row;
    final int num_col_kernel = kernel._num_col;
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
              val += kernel._data[idx_k] * this._data[row_start + rc];
              weight += kernel._data[idx_k];
            }
            ++idx_k;
          }
        }
        if(is_smoothing)
        {
          val /= weight;
        }
        result._data[idx] = val;
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
    assert (mat._num_col == this._num_col);
    assert (mat._num_row == this._num_row);

    final int len = this._data.length;
    for(int i = 0; i < len; ++i)
    {
      this._data[i] = Math.abs(this._data[i]) + Math.abs(mat._data[i]);
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
      final int start = r * this._num_col;
      for(int c = 0; c < this._num_col; ++c)
      {
        this._data[start + c] = val;
      }
    }
    // bottom
    for(int r = this._num_row - 1; r >= this._num_row - 1 - b; --r)
    {
      final int start = r * this._num_col;
      for(int c = 0; c < this._num_col; ++c)
      {
        this._data[start + c] = val;
      }
    }
    // left
    for(int c = 0; c < b; ++c)
    {
      int start = c;
      for(int r = 0; r < this._num_row; ++r)
      {
        this._data[start] = val;
        start += this._num_col;
      }
    }
    // right
    for(int c = this._num_col - 1; c >= this._num_col - 1 - b; --c)
    {
      int start = c;
      for(int r = 0; r < this._num_row; ++r)
      {
        this._data[start] = val;
        start += this._num_col;
      }
    }
  }

  /**
   * set values to [0, 1]
   */
  public void normalizeOnSelf()
  {
    float min = this._data[0];
    float max = this._data[0];
    final int len = this._data.length;
    float scale;
    for(float f : this._data)
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
      this._data[i] = (this._data[i] - min) * scale;
    }
  }

  public static int[] getSortOrder(float[] value)
  {
    return Matrix.getSortOrder(value, true);
  }

  public static int[] getSortOrder(float[] value, boolean is_descending)
  {
    List<Float> list = new ArrayList<>();
    for(float val : value)
    {
      list.add(new Float(val));
    }
    return Matrix.getSortOrder(list, is_descending);
  }

  public static int[] getSortOrder(List<Float> list)
  {
    return Matrix.getSortOrder(list, true);
  }

  public static int[] getSortOrder(List<Float> list, boolean is_descending)
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
      if(is_descending)
      {
        result[id_order] = i.intValue();
      }
      else
      {
        result[size_order - id_order - 1] = i.intValue();
      }
      ++id_order;
    }
    return result;
  }

  public int[][] getLocalMaxima(int radius, float minimal, boolean is_sorted)
  {
    List<Float> maximas = new ArrayList<Float>();
    List<Integer> r_max = new ArrayList<Integer>();
    List<Integer> c_max = new ArrayList<Integer>();

    final int len_r_loop = this._num_row;
    final int len_c_loop = this._num_col;
    for(int r = 0; r < len_r_loop; ++r)
    {
      final int row_start_out = r * this._num_col;
      for(int c = 0; c < len_c_loop; ++c)
      {
        final float val = this._data[row_start_out + c];
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
            final int row_start = this_r * this._num_col;
            for(int lc = -radius; lc <= radius; ++lc)
            {
              final int this_c = c + lc;
              if(this_c >= 0 && this_c < len_c_loop && val < this._data[row_start + this_c])
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
      result[id] = this._data[index[id][0] * this._num_col + index[id][1]];
    }
    return result;
  }

  public void thresholdOnSelf(float thres)
  {
    final int len = this._data.length;
    for(int i = 0; i < len; ++i)
    {
      if(this._data[i] < thres)
      {
        this._data[i] = 0.0f;
      }
      else
      {
        this._data[i] = 1.0f;
      }
    }
  }

  public void erodeOnSelf(int radius, float thres)
  {
    final int nr_1 = this._num_row - radius;
    final int nc_1 = this._num_col - radius;
    float[] copy = Arrays.copyOf(this._data, this._data.length);
    Arrays.fill(this._data, 0.0f);
    for(int r = radius; r < nr_1; ++r)
    {
      final int r_s = r * this._num_col;
      for(int c = radius; c < nc_1; ++c)
      {
        boolean is_all_above = true;
        for(int rr = -radius; rr <= radius && is_all_above; ++rr)
        {
          int rs = (rr + r) * this._num_col + c;
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
          this._data[c + r_s] = copy[c + r_s];
        }
      }
    }
  }

  private static void dispArray(int nr, int nc, float[] d)
  {
    for(int r = 0, rs = 0; r < nr; ++r, rs += nc)
    {
      for(int c = 0; c < nc; ++c)
      {
        System.out.print(d[rs + c] + " ");
      }
      System.out.println();
    }
    System.out.println();
  }

  public void disp()
  {
    Matrix.dispArray(this._num_row, this._num_col, this._data);
  }

  public void invGaussJordanOnSelf()
  {
    assert(this._num_col == this._num_row);
    final int size = this._num_col;
    float[] data = new float[size * size];
    Arrays.fill(data, 0.0f);
    for(int i = 0; i < size; ++i)
    {
      data[i * size + i] = 1.0f;
    }

    // clear lower triangle
    for(int c = 0; c < size; ++c)
    {
      final int cs = c * size + c;
      final int length = size - c;
      // reorder for largest element in column
      int index_max = c;
      float abs_max = Math.abs(this._data[cs]);
      int id = cs + size;
      for(int i = c + 1; i < size; ++i, ++id)
      {
        final float abs = Math.abs(this._data[id]);
        if(abs > abs_max)
        {
          abs_max = abs;
          index_max = i;
        }
      }
      if(index_max != c)
      {
        // swap row index_max and c
        final int start0 = index_max * size;
        final int start1 = c * size;
        // max -> tmp
        float[] tmp = Arrays.copyOfRange(this._data, start0, start0 + size);
        // c ->max
        System.arraycopy(this._data, start1, this._data, start0, size);
        // tmp -> c
        System.arraycopy(tmp, 0, this._data, start1, size);
        // same to data(local variable)
        System.arraycopy(data, start0, tmp, 0, size);
        // c ->max
        System.arraycopy(data, start1, data, start0, size);
        // tmp -> c
        System.arraycopy(tmp, 0, data, start1, size);
      }

      final float val_diag = this._data[cs];
      int rs = cs + size;
      final int row_start = c * size;
      for(int cc = 0; cc < size; ++cc)
      {
        this._data[row_start + cc] /= val_diag;
        data[row_start + cc] /= val_diag;
      }

      float[] delta_data = Arrays.copyOfRange(data, cs, cs + length);
      float[] delta_self = Arrays.copyOfRange(this._data, cs, cs + length);
      for(int r = c + 1; r < size; ++r, rs += size)
      {
        final float scale = this._data[rs];
        for(int cc = 0; cc < length; ++cc)
        {
          data[rs + cc] -= scale * delta_data[cc];
          this._data[rs + cc] -= scale * delta_self[cc];
        }
      }
    }

    // clear upper triangle
    for(int c = size - 1; c > 0; --c)
    {
      final int length = size - c;
      final int start = c * size;
      float[] delta_data = Arrays.copyOfRange(data, start, start + size);
      float[] delta_self = Arrays.copyOfRange(this._data, start, start + size);
      int cs = size * (c - 1) + c;
      for(int r = c - 1; r >= 0; --r, cs -= size)
      {
        final float scale = this._data[cs];
        final int rs = r * size;
        for(int cc = 0; cc < size; ++cc)
        {
          data[rs + cc] -= scale * delta_data[cc];
          this._data[rs + cc] -= scale * delta_self[cc];
        }
      }
    }
    this._data = data;
  }
}
