package Solver;

import java.util.Arrays;

public class SudokuSolver
{
  private int[][] _board;
  private int _block;
  private int _range;

  /**
   * checks whether it is possible to put value to board[row][col]
   *
   * @param row
   * @param col
   * @param value
   * @return
   */
  private boolean isTrialValid(int row, int col, int value)
  {
    if(this._board[row][col] > 0)
    {
      return false;
    }
    else
    {
      final int offset_row = (row / this._block) * this._block;
      final int offset_col = (col / this._block) * this._block;
      for(int i = 0; i < this._range; ++i)
      {
        if(value == this._board[row][i])
        {
          return false;
        }
      }
      for(int i = 0; i < this._range; ++i)
      {
        if(value == this._board[i][col])
        {
          return false;
        }
      }
      for(int i = 0; i < this._block; ++i)
      {
        for(int j = 0; j < this._block; ++j)
        {
          if(value == this._board[offset_row + i][offset_col + j])
          {
            return false;
          }
        }
      }
      return true;
    }
  }

  /**
   * checks whether the game is finished
   *
   * @return
   */
  private boolean isComplete()
  {
    for(int r = 0; r < this._range; ++r)
    {
      for(int c = 0; c < this._range; ++c)
      {
        if(this._board[r][c] < 0)
        {
          return false;
        }
      }
    }
    return true;
  }

  /**
   * fill all blanks with unique possibility(in one iteration)
   *
   * @return
   */
  private int fillUnique()
  {
    int num_filled = 0;
    for(int r = 0; r < this._range; ++r)
    {
      for(int c = 0; c < this._range; ++c)
      {
        if(this._board[r][c] < 0)
        {
          int count_alternative = 0;
          int alternative = -1;
          for(int t = 1; t <= 9; ++t)
          {
            if(this.isTrialValid(r, c, t))
            {
              ++count_alternative;
              alternative = t;
            }
          }
          if(count_alternative == 1)
          {
            this._board[r][c] = alternative;
            ++num_filled;
          }
        }
        else
        {
          ++num_filled;
        }
      }
    }
    return num_filled;
  }

  /**
   * print board on screen, for debug
   */
  private void disp()
  {
    for(int r = 0; r < this._range; ++r)
    {
      for(int c = 0; c < this._range; ++c)
      {
        System.out.print(this._board[r][c] + " ");
      }
      System.out.println("");
    }
    System.out.println("");
  }

  /**
   * try to solve the sudoku
   *
   * @return
   */
  public boolean solve()
  {
    int num_filled = 0;
    int num_round = 0;
    int num_total = this._range * this._range;
    while(num_filled < num_total)
    {
      int num_filled_now = this.fillUnique();
      this.disp(); // TODO del
      ++num_round;
      System.out.println("filled " + num_filled_now + " in round " + num_round);
      if(num_filled_now == num_filled)
      {
        System.out.println("filling with unique possibility is not enough");
        return false;
      }
      else
      {
        num_filled = num_filled_now;
      }
    }
    System.out.println("Finshed");
    return true;
  }

  public SudokuSolver(int[][] mapDigits)
  {
    this._block = 3;
    this._range = this._block * this._block;
    this._board = new int[this._range][this._range];
    for(int i = 0; i < this._range; ++i)
    {
      Arrays.fill(this._board[i], -1);
    }
    for(int[] digit : mapDigits)
    {
      assert (digit.length == 3);
      this._board[digit[0]][digit[1]] = digit[2];
    }
  }
}
