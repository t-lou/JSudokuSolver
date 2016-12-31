package NN;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;

public class DataSource {
	private float[][] data_image;
	private int[] data_label;
	private int dim_in;
	private int dim0, dim1;

	/**
	 * convert four bytes to int
	 * @param bytes
	 * @return
	 */
	private int bytesToInt(byte[] bytes) {
		ByteBuffer buffer = ByteBuffer.wrap(bytes);
		buffer.order(ByteOrder.BIG_ENDIAN);
		return buffer.getInt();
	}
	
	/**
	 * convert (unsigned) bytes to floats(one byte to one float), then normalize the array
	 * @param bytes
	 * @return
	 */
	private float[] normalize(byte[] bytes) {
		float[] result = new float[bytes.length];
		float sum = 0.0f;
		float sum_sq = 0.0f;
		float dev = 0.0f;
		float mean;
		
		for(int id = 0; id < bytes.length; ++id) {
			int val_i = 0xFF & (int)bytes[id];
			float val = 1.0f - ((float)val_i) / 255.0f;
			sum += val;
			sum_sq += val * val;
			result[id] = val;
		}

		mean = sum / (float)bytes.length;
		dev = (float)Math.sqrt(sum_sq / (float)bytes.length - mean * mean);
		
		for(int id = 0; id < bytes.length; ++id) {
			result[id] = (result[id] - mean) / dev;
		}
		
		return result;
	}
	
	/**
	 * convert array of (unsigned) byte to int(one byte to one int)
	 * @param bytes
	 * @return
	 */
	private int[] bytesToInts(byte[] bytes) {
		int[] result = new int[bytes.length];
		for(int i = 0; i < bytes.length; ++i) {
			result[i] = 0xFF & (int)bytes[i];
		}
		return result;
	}
	
	/**
	 * convert (r*c) bytes to (r,c) float, each row normalized
	 * @param bytes
	 * @param len
	 * @param sub_len
	 * @return
	 */
	private float[][] bytesToFloatArray(byte[] bytes, int len, int sub_len) {
		assert(len * sub_len == bytes.length);
		float[][] result = new float[len][sub_len];
		for(int id0 = 0; id0 < len; ++id0) {
			result[id0] = normalize(Arrays.copyOfRange(bytes, sub_len * id0, 
					sub_len * (id0 + 1)));
		}
		return result;
	}
	
	/**
	 * read file to bytes
	 * @param path
	 * @return
	 */
	private byte[] readFile(String path) {
		try {
			return Files.readAllBytes(Paths.get(path));
		} catch (Exception e) {
			return new byte[0];
		}
	}
	
	/**
	 * load one pair of mnist files(one image and one label)
	 * @param path_image
	 * @param path_label
	 */
	public DataSource(String path_image, String path_label) {
		byte[] raw_image = this.readFile(path_image);
		byte[] raw_label = this.readFile(path_label);
		int num_image = bytesToInt(Arrays.copyOfRange(raw_image, 4, 8));
		int num_label = bytesToInt(Arrays.copyOfRange(raw_label, 4, 8));
		assert(num_image == num_label);
		this.dim0 = bytesToInt(Arrays.copyOfRange(raw_image, 8, 12));
		this.dim1 = bytesToInt(Arrays.copyOfRange(raw_image, 12, 16));
		this.dim_in = this.dim0 * this.dim1;
		this.data_label = bytesToInts(Arrays.copyOfRange(raw_label, 8, raw_label.length));
		this.data_image = bytesToFloatArray(Arrays.copyOfRange(raw_image, 16, raw_image.length),
				num_label, this.dim_in);
	}
	
	/**
	 * return id-th image
	 * @param id
	 * @return
	 */
	public float[] getImage(int id) {
		assert(id < this.data_label.length);
		return this.data_image[id];
	}
	
	/**
	 * return id-th label
	 * @param id
	 * @return
	 */
	public int getLabel(int id) {
		assert(id < this.data_label.length);
		return this.data_label[id];
	}
	
	/**
	 * get size of dataset
	 * @return
	 */
	public int getNumData() {
		return this.data_label.length;
	}
	
	/**
	 * get dimensionality of feature(image)
	 * @return
	 */
	public int getDim() {
		return this.dim_in;
	}
	
	public int getDim0() {
		return this.dim0;
	}

	public int getDim1() {
		return this.dim1;
	}
}
