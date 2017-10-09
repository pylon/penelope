defmodule Penelope.NIF do
  @moduledoc """
  NIF wrapper module

  for blas, see http://www.netlib.org/blas/
  """

  alias Penelope.ML.Vector, as: Vector

  @on_load :init

  @doc "module initialization callback"
  @spec init() :: :ok
  def init do
    with :error, {_, message} <- :erlang.load_nif('./priv/penelope', 0) do
      raise message
    end
  end

  @doc "y = ax"
  @spec blas_sscal(a::float, x::Vector.t) ::Vector.t
  def blas_sscal(_a, _x) do
    :erlang.nif_error(:nif_library_not_loaded)
  end

  @doc "z = ax + y"
  #@spec blas_saxpy(a::float, x::Vector.t, y::Vector.t) ::Vector.t
  def blas_saxpy(_a, _x, _y) do
    :erlang.nif_error(:nif_library_not_loaded)
  end
end
