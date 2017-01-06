package ImageProc;

/**
 * Created by tlou on 05.01.17.
 */
public class TableRecognizer
{
  public class Guess
  {
    private float _value;
    private float _possibility;

    public Guess()
    {
      this(0.0f, 0.0f);
    }

    public Guess(float value, float possibility)
    {
      this._value = value;
      this._possibility = possibility;
    }
  }
}
