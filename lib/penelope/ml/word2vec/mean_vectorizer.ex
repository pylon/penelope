defmodule Penelope.ML.Word2vec.MeanVectorizer do
  @moduledoc """
  This module vectorizes a list of tokens using word vectors. Token vectors
  are retrieved from the word2vec index (see index.ex). These are combined
  into a single document vector by taking their vector mean.
  """

  alias Penelope.ML.Vector, as: Vector
  alias Penelope.ML.Word2vec.Index, as: Index

  def fit(context, x, y) do
    {context, x, y}
  end

  def transform(context, x, y) do
    %{index: index = %Index{vector_size: vector_size}} = context
    x_vector = Enum.map(x, &vectorize(&1, index, vector_size))
    {context, x_vector, y}
  end

  defp vectorize(x, index, vector_size) do
    x
    |> Stream.map(&Index.lookup!(index, &1))
    |> Enum.reduce(Vector.zeros(vector_size), &Vector.add/2)
    |> Vector.scale(1 / Enum.count(x))
  end
end
