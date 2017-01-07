package ImageProc;

import NN.Matrix;

import java.util.Arrays;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Created by tlou on 05.01.17.
 */
public class TableRecognizer
{
  private float[][] _hough_maxima_index;
  private float[] _hough_maxima_value;
  private float[][][] _hough_table_lines;

  private static float[] getLineFromPoint(float[] pt0, float[] pt1)
  {
    assert (pt0.length == 2 && pt1.length == 2);
    float[] line = new float[3];
    // ax + by + c = 0 -> a = -dy & b = dx
    line[0] = -(pt0[1] - pt1[1]);
    line[1] = pt0[0] - pt1[0];
    final float len = (float) Math.sqrt((double) (line[0] * line[0] + line[1] * line[1]));
    line[0] /= len;
    line[1] /= len;
    line[2] = -(line[0] * pt0[0] + line[1] * pt0[1]);
    return line;
  }

  private static float getDistPoints(float[] pt0, float[] pt1)
  {
    assert (pt0.length == 2 && pt1.length == 2);
    float dx = pt0[0] - pt1[0];
    float dy = pt0[1] - pt1[1];
    return (float) Math.sqrt((double) (dx * dx + dy * dy));
  }

  private static float getDistLinePoint(float[] point, float[] line)
  {
    assert (line.length == 3 && point.length == 2);
    return point[0] * line[0] + point[1] * line[1] + line[2];
  }

  private static float[] getDistLinePoints(float[][] points, float[] line)
  {
    assert (line.length == 3);
    final int length = points.length;
    float[] dists = new float[points.length];
    for(int i = 0; i < length; ++i)
    {
      dists[i] = Math.abs(TableRecognizer.getDistLinePoint(points[i], line));
    }
    return dists;
  }

  private static int[] getIndexInliers(float[] sequence, float threshold)
  {
    return TableRecognizer.getIndexInliers(sequence, threshold, 0.0f);
  }

  private static int[] getIndexInliers(float[] sequence, float threshold, float offset)
  {
    final int length = sequence.length;
    int[] index = new int[length];
    int num_inlier = 0;
    for(int i = 0; i < length; ++i)
    {
      if(Math.abs(sequence[i] - offset) < threshold)
      {
        index[num_inlier] = i;
        ++num_inlier;
      }
    }
    return Arrays.copyOf(index, num_inlier);
  }

  private static int[] getIndexInliers(float[][] points, float[] line, float threshold)
  {
    return TableRecognizer.getIndexInliers(TableRecognizer.getDistLinePoints(points, line), threshold);
  }

  private static float[] reparameterPoints(float[][] points, float[] line)
  {
    assert (line.length == 3);
    final int length = points.length;
    float[] dists = new float[length];
    final float base_x = line[0] * line[2];
    final float base_y = -line[1] * line[2];
    for(int i = 0; i < length; ++i)
    {
      final float dx = points[i][0] - base_x;
      final float dy = points[i][1] - base_y;
      dists[i] = (float) Math.sqrt((double) (dx * dx + dy * dy));
    }
    return dists;
  }

  private static int[] getInverseIndex(int total, int[] subset)
  {
    boolean[] is_in_other = new boolean[total];
    int num_other = total;
    Arrays.fill(is_in_other, true);
    for(int i : subset)
    {
      assert (i < total);
      if(is_in_other[i]) // avoid duplicate
      {
        --num_other;
        is_in_other[i] = false;
      }
    }
    int[] not_subset = new int[num_other];
    for(int i = 0, j = 0; i < total; ++i)
    {
      if(is_in_other[i])
      {
        not_subset[j] = i;
        ++j;
      }
    }
    return not_subset;
  }

  private static float[][] subsetPoints(float[][] points, int[] index)
  {
    return TableRecognizer.subsetPoints(points, index, false);
  }

  private static float[][] subsetPoints(float[][] points, int[] index, boolean is_inverse)
  {
    final int[] index_subset = is_inverse ? TableRecognizer.getInverseIndex(points.length, index) : index;
    final int length_pixel = points.length;
    final int length = index_subset.length;
    float[][] subset = new float[length][2];
    int j = 0;
    for(int i : index_subset)
    {
      assert ((i < length) && (length_pixel < length_pixel) && (points[i].length == 2));
      subset[j][0] = points[i][0];
      subset[j][1] = points[i][1];
      ++j;
    }
    return subset;
  }

  static class SparseIntArray
  {
    public int _count;
    public int[] _index;
    public int[] _value;

    public SparseIntArray()
    {
    }

    public void resize(int size)
    {
      this._index = new int[size];
      this._value = new int[size];
      this._count = 0;
    }

    public void add(int id, int val)
    {
      assert (this._count < this._value.length);
      this._index[this._count] = id;
      this._value[this._count] = val;
      ++this._count;
    }

    public void finish()
    {
      this._index = Arrays.copyOf(this._index, this._count);
      this._value = Arrays.copyOf(this._value, this._count);
    }
  }

  private static SparseIntArray getIndexUnidistantSequence(float[] sequence)
  {
    final int num_iter = sequence.length * 5;
    final int length = sequence.length;
    float best_score = -1.0f;
    final float thres_close = 5.0f;
    int id_base = -1;
    float unidist = -1.0f;

    for(int iter = 0; iter < num_iter; ++iter)
    {
      final int index0 = ThreadLocalRandom.current().nextInt(length);
      final int index1 = ThreadLocalRandom.current().nextInt(length);
      final float dist = Math.abs(sequence[index0] - sequence[index1]);
      if(dist > thres_close || dist < -thres_close)
      {
        float score = 0.0f;
        boolean is_to_smaller = false;
        for(int id = 0; id < length && !is_to_smaller; ++id)
        {
          final float dist_ = sequence[id] - sequence[index1];
          float mulitple = Math.abs(dist_ / dist);
          if(mulitple > 5.0f)
          {
            is_to_smaller = true;
            break;
          }
          if(mulitple > 0.5f)
          {
            mulitple /= (float) Math.round(mulitple);
          }
          mulitple = mulitple > 1.0f ? 1.0f / mulitple : mulitple;
          score += mulitple;
        }
        if(!is_to_smaller && score > best_score)
        {
          best_score = score;
          id_base = index1;
          unidist = dist;
        }
      }
      else
      {
        // too close, redo
        --iter;
        continue;
      }
    }
    SparseIntArray result = new SparseIntArray();
    if(id_base >= 0)
    {
      float[] distances = new float[length];
      int[] offsets = new int[length];
      result.resize(length);
      for(int i = 0; i < length; ++i)
      {
        distances[i] = sequence[i] - sequence[id_base];
        offsets[i] = Math.round(distances[i] / unidist);
        distances[i] = Math.abs(distances[i] - (float) offsets[i]);
      }
      for(int i = 0; i < length; ++i)
      {
        final boolean is_l_higher = (i > 0) && (offsets[i - 1] == offsets[i])
            && (distances[i - 1] > distances[i]);
        final boolean is_r_higher = (i < length - 2) && (offsets[i + 1] == offsets[i])
            && (distances[i + 1] > distances[i]);
        if(i == id_base || (!is_l_higher && !is_r_higher))
        {
          result.add(i, offsets[i]);
        }
      }
      result.finish();
    }
    return result;
  }

  private static float[] fitLine(float[][] points)
  {
    final int num_iter = points.length * 5;
    final int length = points.length;
    float[] best_line = new float[0];
    int best_num_inlier = 0;
    final float thres_close = 10.0f;

    for(int iter = 0; iter < num_iter; ++iter)
    {
      final int i = ThreadLocalRandom.current().nextInt(length);
      final int j = ThreadLocalRandom.current().nextInt(length);
      if(TableRecognizer.getDistPoints(points[i], points[j]) > thres_close)
      {
        final float[] line = TableRecognizer.getLineFromPoint(points[i], points[j]);
        final int[] index_inliers = TableRecognizer.getIndexInliers(points, line, thres_close);
        final int num_inlier = index_inliers.length;
        if(num_inlier > best_num_inlier)
        {
          best_num_inlier = num_inlier;
          best_line = line;
        }
      }
      else
      {
        --iter;
        continue;
      }
    }

    return best_line;
  }

  private static float[][] getContinuousHoughs(SparseIntArray index, float[][] points)
  {
    float[][] result = new float[0][2];
    final int length = index._value.length;
    for(int id = 3; id < length; ++id)
    {
      if(index._value[id] == index._value[id - 3] + 3)
      {
        // four continuous
        result = new float[4][2];
        for(int i = 0; i < 4; ++i)
        {
          result[i][0] = points[index._index[id + i - 3]][0];
          result[i][1] = points[index._index[id + i - 3]][1];
        }
        return result;
      }
    }
    for(int id = 2; id < length; ++id)
    {
      if(index._value[id] == index._value[id - 2] + 3)
      {
        // one point is missing(0 1 3 or 0 2 3)
        // interpolate two in the middle -> 0 1 2 3
        result = new float[4][2];
        final float x = points[index._index[id - 2]][0];
        final float y = points[index._index[id - 2]][1];
        final float dx = points[index._index[id]][0] / 3.0f;
        final float dy = points[index._index[id]][1] / 3.0f;
        for(int i = 0; i < 4; ++i)
        {
          result[i][0] = x + dx * (float) i;
          result[i][1] = y + dy * (float) i;
        }
        return result;
      }
    }
    return result;
  }

  private static int[] reorder(int[] sequence, int[] order)
  {
    int[] ordered = new int[sequence.length];
    for(int i = 0; i < sequence.length; ++i)
    {
      ordered[i] = order[sequence[i]];
    }
    return ordered;
  }

  public void proceed()
  {
    // local copy of points
    float[][] copy_points = new float[this._hough_maxima_index.length][2];
    float[][][] result = new float[2][4][2];
    for(int i = 0; i < this._hough_maxima_index.length; ++i)
    {
      copy_points[i][0] = this._hough_maxima_index[i][0];
      copy_points[i][1] = this._hough_maxima_index[i][1];
    }

    this._hough_table_lines = new float[0][4][2];
    for(int iter = 0, count_dim = 0; iter < 10 && count_dim < 2; ++iter)
    {
      float[] line = TableRecognizer.fitLine(copy_points);
      int[] index_inliers = TableRecognizer.getIndexInliers(copy_points, line, 10.0f);
      float[][] point_inliers = TableRecognizer.subsetPoints(copy_points, index_inliers);
      float[] params = TableRecognizer.reparameterPoints(point_inliers, line);
      int[] order = Matrix.getSortOrder(params, false);
      {
        float[] copy = params.clone();
        for(int i = 0; i < params.length; ++i)
        {
          params[i] = copy[order[i]];
        }
      }
      SparseIntArray index_unidist_points = TableRecognizer.getIndexUnidistantSequence(params);
      index_unidist_points._index = TableRecognizer.reorder(index_unidist_points._index, order);
      float[][] table_lines = TableRecognizer.getContinuousHoughs(index_unidist_points, point_inliers);
      if(table_lines.length != 0)
      {
        int[] to_remove = TableRecognizer.reorder(index_unidist_points._index, index_inliers);

        copy_points = TableRecognizer.subsetPoints(copy_points, to_remove, true);
        result[count_dim] = table_lines;
        ++count_dim;
        if(count_dim == 2)
        {
          this._hough_table_lines = result;
        }
      }
    }
  }

  public void setHoughPoints(int[][] index, float[] value)
  {
    assert (index.length == value.length);
    final int length = index.length;
    this._hough_maxima_index = new float[length][2];
    this._hough_maxima_value = Arrays.copyOf(value, length);
    for(int i = 0; i < length; ++i)
    {
      assert (index[i].length == 2);
      this._hough_maxima_index[i][0] = (float) index[i][0];
      this._hough_maxima_index[i][1] = (float) index[i][1];
    }
  }

  public float[][][] getTableLines()
  {
    return this._hough_table_lines;
  }

  public boolean isRecognized()
  {
    return this._hough_table_lines != null && this._hough_table_lines.length >= 2;
  }

  public TableRecognizer()
  {
  }
}
