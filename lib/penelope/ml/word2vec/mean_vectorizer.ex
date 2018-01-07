defmodule Penelope.ML.Word2vec.MeanVectorizer do
  @moduledoc """
  This module vectorizes a list of tokens using word vectors. Token vectors
  are retrieved from the word2vec index (see index.ex). These are combined
  into a single document vector by taking their vector mean.
  """

  alias Penelope.ML.Vector, as: Vector
  alias Penelope.ML.Word2vec.Index, as: Index

  def transform(_model, context, x) do
    %{word2vec_index: index = %Index{vector_size: vector_size}} = context
    Enum.map(x, &vectorize(&1, index, vector_size))
  end

  defp vectorize(x, index, vector_size) do
    x
    |> Stream.map(&lookup(&1, index))
    |> Enum.reduce(Vector.zeros(vector_size), &Vector.add/2)
    |> Vector.scale(1 / Enum.count(x))
  end

  defp lookup(x, index) do
    {_id, v} = Index.lookup!(index, x)
    v
  end
end
