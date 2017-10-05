defmodule NLPotion.ML.Word2vec.Index do
  @moduledoc """
  This module represents a word2vec-style vectorset, compiled into a
  set of hash-partitioned DETS files. Each record is a tuple consisting
  of the term (word) and a set of weights (vector). This module also
  supports parsing the standard text representation of word vectors
  via the compile function.
  """

  alias __MODULE__, as: Index

  defstruct tables: {}, partitions: 1

  @type t :: %Index{tables: tuple, partitions: pos_integer}

  @doc """
  creates a new word2vec index

  files will be created as <path>/<name>_<part>.dets, one per partition
  """
  @spec create!(path::String.t,
               name::String.t,
               [partitions: pos_integer,
                size_hint:  pos_integer]) :: Index.t
  def create!(path, name, options) do
    [partitions: partitions, size_hint:  size_hint] = options
    size_hint = div(size_hint, partitions)
    File.mkdir_p!(path)
    tables = 0..partitions - 1
             |> Stream.map(&create_table(name, path, &1, size_hint))
             |> Enum.reduce({}, &Tuple.append(&2, &1))
    %Index{tables: tables, partitions: partitions}
  end

  @doc """
  opens an existing word2vec index at the specified path
  """
  @spec open!(path::String.t) :: Index.t
  def open!(path) do
    files = File.ls!(path)
    tables = files
             |> Stream.map(&Path.join(path, &1))
             |> Stream.map(&open_table/1)
             |> Enum.reduce({}, &Tuple.append(&2, &1))
    %Index{tables: tables, partitions: tuple_size(tables)}
  end

  @doc """
  closes the index
  """
  @spec close(index::Index.t) :: :ok
  def close(%Index{tables: tables}) do
    tables
    |> Tuple.to_list()
    |> Enum.each(&:dets.close/1)
  end

  defp create_table(name, path, partition, size_hint) do
    part = partition
           |> Integer.to_string
           |> String.pad_leading(2, "0")
    name = "#{name}_#{part}"
    file = path
           |> Path.join("#{name}.dets")
           |> String.to_charlist()
    options = [file:         file,
               access:       :read_write,
               type:         :set,
               min_no_slots: size_hint]
    case :dets.open_file(String.to_atom(name), options) do
      {:ok, file}      -> file
      {:error, reason} -> raise IndexError, reason
    end
  end

  defp open_table(file) do
    name = Path.basename(file, ".dets")
    options = [file:   String.to_charlist(file),
               access: :read]
    case :dets.open_file(String.to_atom(name), options) do
      {:ok, file}      -> file
      {:error, reason} -> raise IndexError, reason
    end
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
  @spec parse_insert!(index::Index.t, line::String.t) :: tuple
  def parse_insert!(index, line) do
    record = parse_line!(line)
    insert!(index, record)
    record
  end

  @doc """
  parses a word vector line: "<term> <weight> <weight> ..."
  """
  @spec parse_line!(line::String.t) :: tuple
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
  @spec insert!(index::Index.t, record::tuple) :: :ok
  def insert!(index, record) do
    case index
         |> get_table(elem(record, 0))
         |> :dets.insert(record) do
      :ok              -> :ok
      {:error, reason} -> raise IndexError, reason
    end
  end

  @doc """
  searches for a term in the word2vec index

  if found, returns the word vector tuple (without the term)
  otherwise, returns nil
  """
  @spec lookup!(index::Index.t, term::String.t) :: tuple | nil
  def lookup!(index, term) do
    case index
         |> get_table(term)
         |> :dets.lookup(term) do
      [{_term, weights}] -> weights
      []                 -> nil
      {:error, reason}   -> raise IndexError, reason
    end
  end

  defp get_table(%Index{tables: tables, partitions: partitions}, term) do
    partition = rem(:xxhash.hash32(term), partitions)
    elem(tables, partition)
  end
end

defmodule IndexError do
  @moduledoc "DETS index processing error"

  defexception message: "an index error occurred"
end
