defmodule Penelope.Application do
  @moduledoc """
  This is the library application for the Penelope framework. It starts the
  supervision tree for the library's processes.
  """

  use Application

  import Supervisor.Spec

  @doc """
  starts the application instance
  """
  def start(_type, _args) do
    children = [
      supervisor(Penelope.ML.Registry, [])
    ]

    options = [
      name: Penelope.Supervisor,
      strategy: :one_for_one
    ]

    Supervisor.start_link(children, options)
  end
end
