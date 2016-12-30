package Solver;

import java.io.File;
import javax.imageio.ImageIO;

import ImageProc.ImageProc;

import java.awt.image.BufferedImage;

public class SupremeSolver {
	private Solver solver;
	private ImageProc img_proc;
	private BufferedImage image_original;
	
	public SupremeSolver(String filename) {
		try {
			this.image_original = ImageIO.read(new File(filename));
			System.out.println(this.image_original.getWidth());
			System.out.println(this.image_original.getHeight());
			System.out.println(this.image_original.getType());
		} catch (Exception e) {
			this.image_original = new BufferedImage(0, 0, 0);
		}
	}
	
	
}
