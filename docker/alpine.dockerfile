FROM bitwalker/alpine-elixir:1.5.2

RUN apk --no-cache --update upgrade && \
    apk --no-cache add busybox make g++ linux-headers openblas-dev

ADD mix.exs mix.lock ./
RUN mix do deps.get, deps.compile

ADD c_src ./c_src
ADD lib ./lib
ADD test ./test

ENTRYPOINT ["mix"]
