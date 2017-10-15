defmodule Penelope.NIF do
  @moduledoc """
  NIF wrapper module

  for blas, see http://www.netlib.org/blas/
  for libsvm, see https://github.com/cjlin1/libsvm
  """

  alias Penelope.ML.Vector

  @on_load :init

  @doc "module initialization callback"
  @spec init() :: :ok
  def init do
    with {:error, reason} <- :erlang.load_nif('./priv/penelope', 0) do
      raise inspect(reason)
    end
  end

  @doc "y = ax"
  @spec blas_sscal(a::float, x::Vector.t) ::Vector.t
  def blas_sscal(_a, _x) do
    :erlang.nif_error(:nif_library_not_loaded)
  end

  @doc "z = ax + y"
  @spec blas_saxpy(a::float, x::Vector.t, y::Vector.t) ::Vector.t
  def blas_saxpy(_a, _x, _y) do
    :erlang.nif_error(:nif_library_not_loaded)
  end

  @doc "trains an svm model using libsvm"
  @spec svm_train(x::[Vector.t], y::[integer], params::map) :: reference
  def svm_train(_x, _y, _params) do
    :erlang.nif_error(:nif_library_not_loaded)
  end

  @doc "extracts svm model parameters from a model resource"
  @spec svm_export(model::reference) :: map
  def svm_export(_model) do
    :erlang.nif_error(:nif_library_not_loaded)
  end

  @doc "compiles svm model parameters into a model resource"
  @spec svm_compile(params::map) :: reference
  def svm_compile(_params) do
    :erlang.nif_error(:nif_library_not_loaded)
  end

  @doc "predicts a class from a feature vector"
  @spec svm_predict_class(model::reference, x::Vector.t) :: integer
  def svm_predict_class(_model, _x) do
    :erlang.nif_error(:nif_library_not_loaded)
  end

  @doc "predicts an ordered list of class probabilities from a feature vector"
  @spec svm_predict_probability(model::reference, x::Vector.t) :: [float]
  def svm_predict_probability(_model, _x) do
    :erlang.nif_error(:nif_library_not_loaded)
  end
end
