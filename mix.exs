defmodule Penelope.Mixfile do
  use Mix.Project

  def project do
    [
      app: :penelope,
      version: "0.1.0",
      elixir: "~> 1.5",
      compilers: [:nif | Mix.compilers],
      start_permanent: Mix.env == :prod,
      deps: deps(),
      dialyzer: [ignore_warnings: ".dialyzerignore",
                 plt_add_deps: :transitive]
    ]
  end

  def application do
    [
      extra_applications: [:logger]
    ]
  end

  defp deps do
    [
      {:credo, "~> 0.3", only: [:dev, :test]},
      {:dialyxir, "~> 0.5", only: [:dev], runtime: false},
      {:xxhash, "~> 0.2.0", hex: :erlang_xxhash},
      {:e2qc, "~> 1.2"},
      {:benchee, "~> 0.9", only: :dev},
      {:stream_data, "~> 0.3", only: [:test]}
    ]
  end
end

defmodule Mix.Tasks.Compile.Nif do
  def run(_args) do
    {result, _errcode} = System.cmd("make", ["-s", "-C", "c_src"])
    IO.binwrite(result)
  end
end
