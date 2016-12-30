package Solver;

import java.util.Arrays;

public class SudokuSolver {
	private int[][] board;
	private int block;
	private int range;
	
	/**
	 * checks whether it is possible to put value to board[row][col]
	 * @param row
	 * @param col
	 * @param value
	 * @return
	 */
	private boolean isTrialValid(int row, int col, int value) {
		if(this.board[row][col] > 0) {
			return false;
		}
		else {
			int offset_row = (row / this.block) * this.block;
	        int offset_col = (col / this.block) * this.block;
	        for(int i = 0; i < this.range; ++i) {
	            if(value == this.board[row][i]) {
	                return false;
	            }
	        }
	        for(int i = 0; i < this.range; ++i) {
	            if(value == this.board[i][col]) {
	                return false;
	            }
	        }
	        for(int i = 0; i < this.block; ++i) {
	            for(int j = 0; j < this.block; ++j) {
	                if(value == this.board[offset_row + i][offset_col + j]) {
	                    return false;
	                }
	            }
	        }
			return true;
		}
	}
	
	/**
	 * checks whether the game is finished
	 * @return
	 */
	private boolean isComplete() {
	    for(int r = 0; r < this.range; ++r) {
	        for(int c = 0; c < this.range; ++c) {
	            if(this.board[r][c] < 0) {
	                return false;
	            }
	        }
	    }
	    return true;
	}
	
	/**
	 * fill all blanks with unique possibility(in one iteration)
	 * @return
	 */
	private int fillUnique() {
	    int num_filled = 0;
	    for(int r = 0; r < this.range; ++r) {
	        for(int c = 0; c < this.range; ++c) {
	            if(this.board[r][c] < 0) {
	            	int count_alternative = 0;
	            	int alternative = -1;
	                for(int t = 1; t <= 9; ++t) {
	                    if(this.isTrialValid(r, c, t)) {
	                        ++count_alternative;
	                        alternative = t;
	                    }
	                }
	                if(count_alternative == 1) {
	                    this.board[r][c] = alternative;
	                    ++num_filled;
	                }
	            }
	            else {
	                ++num_filled;
	            }
	        }
	    }
	    return num_filled;
	}
	
	/**
	 * print board on screen, for debug
	 */
	private void disp() {
	    for(int r = 0; r < this.range; ++r) {
	        for(int c = 0; c < this.range; ++c) {
	        	System.out.print(this.board[r][c] + " ");
	        }
	        System.out.println("");
	    }
        System.out.println("");
	}
	
	/**
	 * try to solve the sudoku
	 * @return
	 */
	public boolean solve() {
		int num_filled = 0;
		int num_round = 0;
		int num_total = this.range * this.range;
		while(num_filled < num_total) {
	        int num_filled_now = this.fillUnique();
	        this.disp(); // TODO del
	        ++num_round;
        	System.out.println("filled " + num_filled_now + " in round " + num_round);
	        if(num_filled_now == num_filled) {
	        	System.out.println("filling with unique possibility is not enough");
	            return false;
	        }
	        else {
	            num_filled = num_filled_now;
	        }
	    }
    	System.out.println("Finshed");
    	return true;
	}
	
	public SudokuSolver(int[][] mapDigits) {
		this.block = 3;
		this.range = this.block * this.block;
		this.board = new int[this.range][this.range];
		for(int i = 0; i < this.range; ++i) {
			Arrays.fill(this.board[i], -1);
		}
		for(int[] digit : mapDigits) {
			assert(digit.length == 3);
			this.board[digit[0]][digit[1]] = digit[2];
		}
	}
}
