defmodule Penelope.ML.Vector do
  @moduledoc """
  This is a the vector library used by the ML modules. It provides an
  interface to an efficient binary representation of 32-bit floating point
  values. Math is done via the BLAS interface, wrapped in a NIF module.
  """

  alias Penelope.NIF, as: NIF

  @type t::binary

  @doc "the empty vector"
  @spec empty() :: t
  def empty do
    <<>>
  end

  @doc "calculates the number of elements in a vector"
  @spec size(vector::t) :: non_neg_integer
  def size(vector) do
    div(byte_size(vector), 4)
  end

  @doc "retrieves a vector element by 0-based index"
  @spec get(vector::t, index::non_neg_integer) :: float
  def get(vector, index) do
    vector
    |> binary_part(index * 4, 4)
    |> binary2float()
  end

  @doc "creates a vector of length n containing all zeros"
  @spec zeros(n::non_neg_integer) :: t
  def zeros(0), do: empty()
  def zeros(n) do
    from_list(for _ <- 1..n, do: 0)
  end

  @doc "converts a list of floats to a vector"
  @spec from_list(numbers::[float]) :: t
  def from_list(numbers) do
    numbers
    |> Enum.map(&float2binary/1)
    |> Enum.reduce(empty(), &(&2 <> &1))
  end

  @doc "converts a vector to a list of floats"
  @spec to_list(vector::t) :: [float]
  def to_list(<<>>), do: []
  def to_list(vector) do
    Enum.map(0..size(vector) - 1, &get(vector, &1))
  end

  @doc "computes y = ax"
  @spec scale(x::t, a::float) :: t
  def scale(x, a), do: NIF.blas_sscal(a / 1, x)

  @doc "computes z = x + y"
  @spec add(x::t, y::t) :: t
  def add(x, y), do: NIF.blas_saxpy(1.0, x, y)

  @doc "computes z = ax + y"
  @spec scale_add(y::t, a::float, x::t) :: t
  def scale_add(y, a, x), do: NIF.blas_saxpy(a / 1, x, y)

  defp binary2float(<<value::float()-native()-size(32)>>), do: value
  defp binary2float(_value), do: :NaN

  defp float2binary(:NaN), do: <<0, 0, 128, 127>>
  defp float2binary(x), do: <<x::float()-native()-size(32)>>
end
