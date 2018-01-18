defmodule Penelope.ML.Registry do
  @moduledoc """
  The ML pipeline registry decouples the names of pipeline components from
  their module names, so that modules can be refactored without breaking
  stored models. The built-in Penelope components are registered automatically,
  but custom components can be added via the `register` function.

  Inverse lookups are also supported for exporting compiled models. The
  registry falls back on module atoms for unregistered components.
  """

  @agent __MODULE__
  @defaults %{
    feature_stack: Penelope.ML.Feature.StackVectorizer,
    feature_merge: Penelope.ML.Feature.MergeFeaturizer,
    context_featurizer: Penelope.ML.Feature.ContextFeaturizer,
    lowercase_preprocessor: Penelope.ML.Text.LowercasePreprocessor,
    ptb_tokenizer: Penelope.ML.Text.PTBTokenizer,
    ptb_digit_tokenizer: Penelope.ML.Text.PTBDigitTokenizer,
    count_vectorizer: Penelope.ML.Text.CountVectorizer,
    regex_vectorizer: Penelope.ML.Text.RegexVectorizer,
    token_featurizer: Penelope.ML.Text.TokenFeaturizer,
    word2vec_mean_vectorizer: Penelope.ML.Word2vec.MeanVectorizer,
    linear_classifier: Penelope.ML.Linear.Classifier,
    svm_classifier: Penelope.ML.SVM.Classifier,
    crf_tagger: Penelope.ML.CRF.Tagger
  }

  @doc """
  starts the registry process
  """
  @spec start_link() :: {:ok, pid} | {:error, any}
  def start_link do
    Agent.start_link(fn -> @defaults end, name: @agent)
  end

  @doc """
  adds a new alias for a pipeline component to the registry
  """
  @spec register(name :: String.t() | atom, module :: atom) :: :ok
  def register(name, module) when is_binary(name) do
    register(String.to_atom(name), module)
  end

  def register(name, module) do
    Agent.update(@agent, fn s -> Map.put(s, name, module) end)
  end

  @doc """
  locates a pipeline component module from its name, falling back on the
  module itself
  """
  @spec lookup(name :: String.t() | atom) :: atom
  def lookup(name) when is_binary(name) do
    lookup(String.to_atom(name))
  end

  def lookup(name) do
    module =
      @agent
      |> Agent.get(& &1)
      |> Map.get(name, name)

    case Code.ensure_loaded(module) do
      {:module, module} -> module
      _ -> raise ArgumentError, message: "invalid name #{name}"
    end
  end

  @doc """
  performs a module->name reverse lookup
  """
  @spec invert(module :: atom) :: String.t()
  def invert(module) do
    @agent
    |> Agent.get(& &1)
    |> Map.new(fn {k, v} -> {v, k} end)
    |> Map.get(module, module)
    |> to_string()
  end
end
