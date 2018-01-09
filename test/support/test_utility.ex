defmodule Penelope.TestUtility do
  @moduledoc """
  test helper utilities
  """

  import ExUnit.Assertions

  @doc """
  32-bit float comparison within tolerance
  """
  def float_equals(a, b) do
    abs(a - b) < 1.0e-5
  end

  @doc """
  just asserts that something was thrown
  """
  def assert_raise(f) do
    f.()
  else
    _ -> assert false, "expected: exception"
  rescue
    _ -> :ok
  end
end
