defmodule NLPotion.ML.Word2vec.Index do
  @moduledoc """
  This module represents a word2vec-style vectorset, compiled into a
  set of hash-partitioned DETS files. Each record is a tuple consisting
  of the term (word) and a set of weights (vector). This module also
  supports parsing the standard text representation of word vectors
  via the compile function.

  On disk, the following files are created:
    <path>/header.dets          index header (version, metadata)
    <path>/<name>_<part>.dets   partition file
  """

  alias __MODULE__, as: Index

  defstruct version: 1, partitions: 1, vector_size: 300, tables: []

  @type t :: %Index{version:     pos_integer,
                    partitions:  pos_integer,
                    vector_size: pos_integer,
                    tables:      [atom]}
  @version 1

  @doc """
  creates a new word2vec index

  files will be created as <path>/<name>_<part>.dets, one per partition
  """
  @spec create!(path::String.t,
                name::String.t,
                [partitions:  pos_integer,
                 size_hint:   pos_integer,
                 vector_size: pos_integer]) :: Index.t
  def create!(path, name, options \\ []) do
    partitions  = Keyword.get(options, :partitions, 1)
    vector_size = Keyword.get(options, :vector_size, 300)
    size_hint   = div(Keyword.get(options, :size_hint, 200_000), partitions)
    header = [version:     @version,
              name:        name,
              partitions:  partitions,
              vector_size: vector_size]

    File.mkdir_p!(path)
    create_header(path, header)
    tables = 0..partitions - 1
             |> Stream.map(&create_table(path, name, &1, size_hint))
             |> Enum.reduce([], &(&2 ++ [&1]))

    %Index{version:     @version,
           partitions:  partitions,
           vector_size: vector_size,
           tables:      tables}
  end

  defp create_header(path, header) do
    file = path
           |> Path.join("header.dets")
           |> String.to_charlist()
    options = [file:         file,
               access:       :read_write,
               type:         :set,
               min_no_slots: 1]
    with {:ok, file} <- :dets.open_file(:word2vec_header, options),
         :ok         <- :dets.insert(file, {:header, header}) do
      :dets.close file
    else
      {:error, reason} -> raise IndexError, reason
    end
  end

  defp create_table(path, name, partition, size_hint) do
    {name, file} = table_file(path, name, partition)
    options = [file:         file,
               access:       :read_write,
               type:         :set,
               min_no_slots: size_hint]
    case :dets.open_file(name, options) do
      {:ok, file}      -> file
      {:error, reason} -> raise IndexError, reason
    end
  end

  defp table_file(path, name, partition) do
    part = partition
           |> Integer.to_string()
           |> String.pad_leading(2, "0")
    name = "#{name}_#{part}"
    file = path
           |> Path.join("#{name}.dets")
           |> String.to_charlist()
    {String.to_atom(name), file}
  end

  @doc """
  opens an existing word2vec index at the specified path
  """
  @spec open!(path::String.t) :: Index.t
  def open!(path) do
    [version:     version,
     name:        name,
     partitions:  partitions,
     vector_size: vector_size] = open_header(path)
    tables = 0..partitions - 1
             |> Stream.map(&open_table(path, name, &1))
             |> Enum.reduce([], &(&2 ++ [&1]))
    %Index{version:     version,
           partitions:  partitions,
           vector_size: vector_size,
           tables:      tables}
  end

  defp open_header(path) do
    file = path
           |> Path.join("header.dets")
           |> String.to_charlist()
    options = [file:   file,
               access: :read,
               type:   :set]
    with {:ok, file}         <- :dets.open_file(:word2vec_header, options),
         [{:header, header}] <- :dets.lookup(file, :header) do
      :dets.close file
      header
    else
      {:error, reason} -> raise IndexError, reason
    end
  end

  defp open_table(path, name, partition) do
    {name, file} = table_file(path, name, partition)
    case :dets.open_file(name, file: file, access: :read) do
      {:ok, file}      -> file
      {:error, reason} -> raise IndexError, reason
    end
  end

  @doc """
  closes the index
  """
  @spec close(index::Index.t) :: :ok
  def close(%Index{tables: tables}) do
    Enum.each(tables, &:dets.close/1)
  end

  @doc """
  inserts word vectors from a text file into a word2vec index

  the index must have been opened using create()
  """
  @spec compile!(index::Index.t, path::String.t) :: :ok
  def compile!(index, path) do
    path
    |> File.stream!()
    |> Task.async_stream(&parse_insert!(index, &1), ordered: false)
    |> Stream.run()
  end

  @doc """
  parses and inserts a single word vector text line into a word2vec index
  """
  @spec parse_insert!(index::Index.t, line::String.t) :: {String.t, binary}
  def parse_insert!(index, line) do
    record = parse_line!(line)
    insert!(index, record)
    record
  end

  @doc """
  parses a word vector line: "<term> <weight> <weight> ..."
  """
  @spec parse_line!(line::String.t) :: {String.t, binary}
  def parse_line!(line) do
    [term | weights] = String.split(line, " ")
    weights
    |> Stream.map(&parse_weight/1)
    |> Enum.reduce({term, <<>>}, fn w, {t, ws} -> {t, ws <> w} end)
  end

  defp parse_weight(str) do
    case Float.parse(str) do
      {value, _remain} -> <<value::float-little-size(32)>>
      :error           -> raise ArgumentError, "invalid weight: #{str}"
    end
  end

  @doc """
  inserts a word vector tuple into a word2vec index
  """
  @spec insert!(index::Index.t, record::{String.t, binary}) :: :ok
  def insert!(%Index{vector_size: vector_size} = index,
              {_term, vector} = record) do
    actual_size = div(byte_size(vector), 4)
    unless actual_size === vector_size do
      raise IndexError, "invalid vector size: #{actual_size} != #{vector_size}"
    end
    case index
         |> get_table(elem(record, 0))
         |> :dets.insert(record) do
      :ok              -> :ok
      {:error, reason} -> raise IndexError, reason
    end
  end

  @doc """
  searches for a term in the word2vec index

  if found, returns the word vector (no term)
  otherwise, returns nil
  """
  @spec lookup!(index::Index.t, term::String.t) :: binary
  def lookup!(%{vector_size: vector_size} = index, term) do
    bit_size = vector_size * 32
    case index
         |> get_table(term)
         |> :dets.lookup(term) do
      [{_term, vector}] -> vector
      []                -> <<0::size(bit_size)>>
      {:error, reason}  -> raise IndexError, reason
    end
  end

  defp get_table(%Index{tables: tables, partitions: partitions}, term) do
    partition = rem(:xxhash.hash32(term), partitions)
    Enum.at(tables, partition)
  end
end

defmodule IndexError do
  @moduledoc "DETS index processing error"

  defexception message: "an index error occurred"
end
