defmodule Penelope.ML.SVM.Classifier do
  @moduledoc """
  The SVM classifier uses libsvm for multi-class classification. It provides
  support for training a model, compiling/extracting model parameters to/from
  erlang data structures, and predicting classes or probabilities.

  Features are represented as lists of dense Vector instances. Classes are
  represented as integers, and class labels for training are lists of these.

  Model parameters are elixir analogs of those supported by libsvm. See
  https://github.com/cjlin1/libsvm for details.
  """

  alias Penelope.ML.Vector
  alias Penelope.NIF

  @doc """
  trains an SVM model and adds it to the pipeline context map

  |key           |description                               |default  |
  |--------------|------------------------------------------|---------|
  |`kernel`      |one of `:linear`/`:rbf`/`:poly`/`:sigmoid`|`:linear`|
  |`degree`      |polynomial degree                         |3        |
  |`gamma`       |training example reach - `:auto` for 1/N  |`:auto`  |
  |`coef0`       |independent term                          |0.0      |
  |`c`           |error term penalty                        |a        |
  |`weights`     |class weights map - `:auto` for balanced  |`:auto`  |
  |`epsilon`     |tolerance for stopping                    |0.001    |
  |`cache_size`  |kernel cache size, in MB                  |1        |
  |`shrinking?`  |use the shrinking heuristic?              |true     |
  |`probability?`|enable class probabilities?               |false    |
  """
  @spec fit(context::map, x::[Vector.t], y::[integer], options::keyword)
    :: {map, [Vector.t], [integer]}
  def fit(context, x, y, options \\ []) do
    if length(x) !== length(y), do: raise ArgumentError, "mismatched x/y"

    params = fit_params(x, y, options)
    model = NIF.svm_train(x, y, params)
    {Map.put(context, :svm_model, model), x, y}
  end

  @doc """
  extracts model parameters from the pipeline context map

  These parameters are simple elixir objects and can later be passed to
  `compile` to add the model back to the context.
  """
  @spec inspect(%{svm_model: reference}) :: map
  def inspect(%{svm_model: svm_model}) do
    NIF.svm_inspect(svm_model)
  end

  @doc """
  compiles a pre-trained model and adds it to the pipeline context
  """
  @spec compile(context::map, params::map) :: map
  def compile(context, params) do
    model = NIF.svm_compile(params)
    Map.put(context, :svm_model, model)
  end

  @doc """
  predicts a target class from a feature vector
  """
  @spec predict_class(%{svm_model: reference}, x::Vector.t) :: integer
  def predict_class(%{svm_model: model}, x) do
    NIF.svm_predict_class(model, x)
  end

  @doc """
  predicts probabilities for all classes from a feature vector

  The results are returned in a map of `%{label => probability}`.
  """
  @spec predict_probability(%{svm_model: reference}, x::Vector.t)
    :: %{integer => float}
  def predict_probability(%{svm_model: model}, x) do
    Enum.into NIF.svm_predict_probability(model, x), %{}
  end

  defp fit_params(x, y, options) do
    gamma = with :auto <- Keyword.get(options, :gamma, :auto) do
      auto_gamma(x)
    end
    weights = with :auto <- Keyword.get(options, :weights, :auto) do
      auto_weights(y)
    end
    %{
      kernel:       Keyword.get(options, :kernel, :linear),
      degree:       Keyword.get(options, :degree, 3),
      gamma:        gamma / 1,
      coef0:        Keyword.get(options, :coef0, 0) / 1,
      c:            Keyword.get(options, :c, 1) / 1,
      weights:      weights,
      epsilon:      Keyword.get(options, :epsilon, 1.0e-3) / 1,
      cache_size:   Keyword.get(options, :cache_size, 1) / 1,
      shrinking?:   Keyword.get(options, :shrinking?, true),
      probability?: Keyword.get(options, :probability?, false)
    }
  end

  defp auto_gamma([x | _]) do
    1.0 / Vector.size(x)
  end

  defp auto_weights(y) do
    # class frequencies, sample count, and class count
    f = Enum.reduce(y, %{}, &Map.update(&2, &1, 1, fn f -> f + 1 end))
    m = Enum.sum(Map.values(f))
    k = Enum.count(f)

    # weight = samples / (classes * frequency)
    f
    |> Map.keys()
    |> Enum.reduce(f, &Map.update!(&2, &1, fn f -> m / (k * f) end))
  end
end
