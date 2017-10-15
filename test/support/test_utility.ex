defmodule Penelope.TestUtility do
  @moduledoc """
  test helper utilities
  """

  import ExUnit.Assertions

  alias StreamData, as: Gen

  @doc """
  find the index of the largest value in a list
  """
  def argmax(list) do
    list
    |> Enum.with_index()
    |> Enum.reduce(fn {v, i}, {v_max, i_max} ->
          v > v_max && {v, i} || {v_max, i_max}
        end)
    |> elem(1)
  end

  @doc """
  stream generator for strictly positive random floats
  """
  def gen_pos_float do
    gen_non_neg_float()
    |> Gen.map(fn f -> f + 1.0e-6 end)
  end

  @doc """
  stream generator for positive or 0.0 random floats
  """
  def gen_non_neg_float do
    gen_float()
    |> Gen.map(fn f -> abs(f) end)
  end

  @doc """
  stream generator for random floats
  """
  def gen_float do
    {Gen.integer(), Gen.uniform_float()}
    |> Gen.tuple()
    |> Gen.map(fn {i, f} -> i * f end)
  end

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
