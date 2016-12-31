package ImageProc;

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.Arrays;

import javax.imageio.ImageIO;

import NN.Matrix;

public class ImageProc {
	private Matrix image_gray;
	
	/**
	 * save image to file for debugging
	 * @param mat
	 * @param filename
	 * @param scale
	 */
	private void saveImage(Matrix mat, String filename, float scale) {
		BufferedImage image = new BufferedImage(mat.getNumCol(), 
				mat.getNumRow(), BufferedImage.TYPE_3BYTE_BGR);
		for(int r = 0; r < mat.getNumRow(); ++r) {
			int row_start = mat.getNumCol() * r;
			for(int c = 0; c < mat.getNumCol(); ++c) {
				int val = (int)(255.0f * mat.getData()[row_start + c] * scale);
				if(val > 255) {
					val = 255;
				}
				else if(val < 0) {
					val = 0;
				}
				image.setRGB(c, r, new Color(val, val, val).getRGB());
			}
		}
		File ouptut = new File(filename);
		try {
			ImageIO.write(image, "png", ouptut);
		} catch (Exception e) {}
	}
	
	/**
	 * return hough transform
	 * @param image
	 * @return
	 */
	public Matrix transformHough(Matrix image) {
		float[] coss = new float[180];
		float[] sins = new float[180];
		for(int the = 0; the < 180; ++the) {
			double rad = Math.toRadians((double)the);
			coss[the] = (float)Math.cos(rad);
			sins[the] = (float)Math.sin(rad);
		}
		
		int max_r = 0;
		int min_r = 0;
		for(int row = 0; row < image.getNumRow(); row += 5) {
			for(int col = 0; col < image.getNumCol(); col += 5) {
				for(int the = 0; the < 180; the += 10) {
					int rv = Math.round((float)col * coss[the] + (float)row * sins[the]);
					if(max_r < rv) {
						max_r = rv;
					}
					if(min_r > rv) {
						min_r = rv;
					}
				}
			}
		}
		
		float[] hough_values = new float[(max_r - min_r) * 180];
		Arrays.fill(hough_values, 0.0f);
		
		int index = 0;
		for(int row = 0; row < image.getNumRow(); ++row) {
			for(int col = 0; col < image.getNumCol(); ++col) {
				float val = image.getData()[index];
				for(int the = 0; the < 180; ++the) {
					int rv = Math.round((float)col * coss[the] + (float)row * sins[the]);
					if(rv < max_r && rv > min_r) {
						hough_values[(rv - min_r) * 180 + the] += val;
					}
				}
				++index;
			}
		}
		
		return new Matrix(max_r - min_r, 180, hough_values);
	}
	
	/**
	 * some filters needed here
	 */
	public void filter() {
//		this.image_gray = image_gray.conv(new Matrix(3, 3, new float[]{
//				0.077847f, 0.123317f, 0.077847f, 
//				0.123317f, 0.195346f, 0.123317f, 
//				0.077847f, 0.123317f, 0.077847f}));
		Matrix sobel_h = image_gray.conv(new Matrix(3, 3, new float[]{
				1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, -1.0f, -1.0f, -1.0f}));
		// sobel vertical
		this.image_gray = image_gray.conv(new Matrix(3, 3, new float[]{
				1.0f, 0.0f, -1.0f, 1.0f, 0.0f, -1.0f, 1.0f, 0.0f, -1.0f}));
		// sum of absolute value
		this.image_gray.addAbsElemWiseOnSelf(sobel_h);
		this.image_gray.setBoundary(0.0f);
//		this.image_gray.normalizeOnSelf();
		Matrix hough = this.transformHough(this.image_gray);
		hough.normalizeOnSelf();
		this.saveImage(hough, "/tmp/hough.png", 1.0f);
		this.saveImage(this.image_gray, "/tmp/sobel.png", 1.0f);
		hough = hough.conv(new Matrix(3, 3, new float[]{
				0.077847f, 0.123317f, 0.077847f, 
				0.123317f, 0.195346f, 0.123317f, 
				0.077847f, 0.123317f, 0.077847f}));
		hough.normalizeOnSelf();
		this.saveImage(hough, "/tmp/hough_smooth.png", 1.0f);
		int[][] index_max = hough.getLocalMaxima();
	}

	/**
	 * set input image
	 * @param image
	 */
	public void setImage(BufferedImage image) {
		float[] data = new float[image.getHeight() * image.getWidth()];
		for(int r = 0; r < image.getHeight(); ++r) {
			int row_start = r * image.getWidth();
			for(int c = 0; c < image.getWidth(); ++c) {
				Color color = new Color(image.getRGB(c, r));
				data[row_start + c] = (0.299f * (float)color.getRed()
						+ 0.114f * (float)color.getBlue() + 0.587f * (float)color.getGreen()) / 255.0f;
			}
		}
		this.image_gray = new Matrix(image.getHeight(), image.getWidth(), data);
	}
}
