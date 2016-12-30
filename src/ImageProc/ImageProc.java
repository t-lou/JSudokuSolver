package ImageProc;

import java.awt.Color;
import java.awt.image.BufferedImage;

import NN.Matrix;

public class ImageProc {
	private Matrix image_gray;
	
	public void filter() {
		this.image_gray = image_gray.conv(new Matrix(3, 3, new float[]{
				0.077847f, 0.123317f, 0.077847f, 
				0.123317f, 0.195346f, 0.123317f, 
				0.077847f, 0.123317f, 0.077847f}));
		Matrix sobel_h = image_gray.conv(new Matrix(3, 3, new float[]{
				1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, -1.0f, -1.0f, -1.0f}));
		// sobel vertical
		this.image_gray = image_gray.conv(new Matrix(3, 3, new float[]{
				1.0f, 0.0f, -1.0f, 1.0f, 0.0f, -1.0f, 1.0f, 0.0f, -1.0f}));
		// sum of absolute value
		this.image_gray.addAbsElemWiseOnSelf(sobel_h);
	}

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
