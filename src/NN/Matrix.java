package NN;

import java.util.Arrays;
import java.util.Random;

public class Matrix {
	private int num_row, num_col;
	private float[] data;
	
	public Matrix() {
		this(0, 0, false);
	}

	public Matrix(int num_row, int num_col) {
		this(num_row, num_col, false);
	}

	public Matrix(int num_row, int num_col, boolean is_random) {
		this.data = new float[num_row * num_col];
		this.num_col = num_col;
		this.num_row = num_row;
		if(is_random) {
			Random rand = new Random();
			float range = (float)(Math.sqrt(6.0f) / Math.sqrt((float)(num_col + num_row)));
			for(int i = 0; i < this.data.length; ++i) {
				this.data[i] = (rand.nextFloat() - 0.5f) * range;
			}
		}
	}

	public Matrix(int num_row, int num_col, float[] values) {
		this(num_row, num_col, values, false);
	}

	public Matrix(int num_row, int num_col, float[] values, boolean is_add_bias) {
		if(is_add_bias) {
			this.num_row = num_row + 1;
		}
		else {
			this.num_row = num_row;
		}
		this.num_col = num_col;
		this.data = new float[this.num_col * this.num_row];
		Arrays.fill(this.data, 1.0f);
		System.arraycopy(values, 0, this.data, 0, num_row * num_col);
	}
	
	/**
	 * multiply another element-wise on this
	 * @param another
	 */
	public void multElemWiseOnSelf(Matrix another) {
		assert(this.num_row == another.num_row);
		assert(this.num_col == another.num_col);
		for(int i = 0; i < this.data.length; ++i) {
			this.data[i] *= another.data[i];
		}
	}
	
	private int[] genBlockPlan(int len, int blk) {
		int num_blk = len / blk;
		int[] plan;
		if(num_blk > 0)
		{
			int rest = len - num_blk * blk;
			plan = new int[num_blk];
			Arrays.fill(plan, blk);
			plan[num_blk - 1] += rest;
		}
		else {
			plan = new int[1];
			plan[0] = len;
		}
		return plan;
	}
	
	/**
	 * suppose this is r-length vector, vec_b is c-length vector, 
	 *     return (r,c) matrix of this*vec_b.T
	 * @param vec_b
	 * @return
	 */
	public Matrix ger(Matrix vec_b) {
		assert(this.num_col == 1);
		assert(vec_b.num_col == 1);
		Matrix result = new Matrix(this.num_row, vec_b.num_row);
		int blk = 32;
		int[] plan_r = this.genBlockPlan(this.num_row, blk);
		int[] plan_c = this.genBlockPlan(vec_b.num_row, blk);
		int r_offset = 0;
		for(int rb : plan_r) {
			int c_offset = 0;
			for(int cb : plan_c) {
				for(int ir = 0; ir < rb; ++ir) {
					int r = r_offset + ir;
					int row_start = r * vec_b.num_row;
					for(int ic = 0; ic < cb; ++ic) {
						int c = c_offset + ic;
						result.data[row_start + c] = this.data[r] * vec_b.data[c]; 
					}
				}
				c_offset += cb;
			}
			r_offset += rb;
		}
		return result;
	}
	
	/**
	 * add delta on this, this := this + scale * delta
	 * @param delta
	 * @param scale
	 */
	public void addOnSelf(Matrix delta, float scale) {
		assert(this.num_col == delta.num_col);
		assert(this.num_row == delta.num_row);
		for(int i = 0; i < this.data.length; ++i) {
			this.data[i] += delta.data[i] * scale;
		}
	}
	
	/**
	 * return this * another
	 * @param another
	 * @return
	 */
	public Matrix mult(Matrix another) {
		int num_row = this.num_row;
		int num_col = another.getNumCol();
		int num_dua = another.getNumRow();
		assert(this.num_col == num_dua);
		
		float[] values = new float[num_row * num_col];
		Arrays.fill(values, 0.0f);
		// TODO optimize
		if(num_row < 20 || num_col < 20) {
			// naive matrix multiplication
			for(int r = 0; r < num_row; ++r) {
				int row_start = r * num_dua;
				for(int c = 0; c < num_col; ++c) {
					float val = 0.0f;
					for(int i = 0; i < num_dua; ++i) {
						val += this.data[row_start + i] * another.data[i * num_col + c];
					}
					values[r * num_col + c] = val;
				}
			}
		}
		else {
			// blocked matrix multiplication
			int blk = 32;
			int[] plan_r = this.genBlockPlan(num_row, blk);
			int[] plan_c = this.genBlockPlan(num_col, blk);
			int[] plan_d = this.genBlockPlan(num_dua, blk);
			int r_offset = 0;
			for(int rb : plan_r) {
				int c_offset = 0;
				for(int cb : plan_c) {
					for(int ir = 0; ir < rb; ++ir) {
						int r = r_offset + ir;
						for(int ic = 0; ic < cb; ++ic) {
							int c = c_offset + ic;
							int d_offset = 0;
							float delta = 0.0f;
							for(int db : plan_d) {
								for(int id = 0; id < db; ++id) {
									int d = d_offset + id;
									delta += this.data[r * num_dua + d] 
											* another.data[d * num_col + c];
								}
								d_offset += db;
							}
							values[r * num_col + c] += delta;
						}
					}
					c_offset += cb;
				}
				r_offset += rb;
			}
		}
		
		return new Matrix(num_row, num_col, values);
	}
	
	/**
	 * return relu of this
	 * @return
	 */
	public Matrix relu() {
		Matrix result = new Matrix(this.num_row, this.num_col, this.data);
		for(int i = 0; i < result.data.length; ++i) {
			if(result.data[i] < 0.0f) {
				result.data[i] = 0.0f;
			}
		}
		return result;
	}
	
	/**
	 * return derivative of relu of this
	 * @return
	 */
	public Matrix relu_der() {
		Matrix result = new Matrix(this.num_row, this.num_col, this.data);
		for(int i = 0; i < result.data.length; ++i) {
			if(result.data[i] >= 0.0f) {
				result.data[i] = 1.0f;
			}
			else {
				result.data[i] = 0.0f;
			}
		}
		return result;
	}
	
	/**
	 * this := softmax(this), suppose this is vector (1 column)
	 */
	public void softmaxAsVecOnSelf() {
		assert(this.num_row == 1 || this.num_col == 1);
		float sum = 0.0f;
		for(int i = 0; i < this.data.length; ++i) {
			this.data[i] = (float)Math.exp(this.data[i]);
			sum += this.data[i];
		}
		for(int i = 0; i < this.data.length; ++i) {
			this.data[i] /= sum;
		}
	}
	
	/**
	 * this := this.T, suppose this is vector (1 column)
	 */
	public void conjAsVecOnSelf() {
		assert(this.num_row == 1 || this.num_col == 1);
		int num = this.num_row;
		this.num_row = this.num_col;
		this.num_col = num;
	}
	
	/**
	 * this(row, col) := this(row, col) + delta
	 * @param row
	 * @param col
	 * @param delta
	 */
	public void addElement(int row, int col, float delta) {
		assert(row < this.num_row);
		assert(col < this.num_col);
		this.data[row * this.num_col + col] += delta;
	}
	
	/**
	 * this := (this, 1), suppose this is vector (1 column)
	 * @return
	 */
	public Matrix appendAsVecBias() {
		assert(this.num_col == 1);
		return new Matrix(this.num_row, this.num_col, this.data, true);
	}
	
	/**
	 * return data(float[])
	 * @return
	 */
	public float[] getData() {
		return this.data;
	}
	
	/**
	 * return number of columns
	 * @return
	 */
	public int getNumCol() {
		return this.num_col;
	}
	
	/**
	 * return number of rows
	 * @return
	 */
	public int getNumRow() {
		return this.num_row;
	}
}
