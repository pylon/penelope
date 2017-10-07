ERLANG_PATH ?= $(shell erl -eval 'io:format("~s", [lists:concat([code:root_dir(), "/erts-", erlang:system_info(version), "/include"])])' -s init stop -noshell)
CC = gcc
CFLAGS += -Wall -O3 -flat_namespace -undefined suppress -fpic -shared \
          -I$(ERLANG_PATH) -lblas
OUTDIR = ../priv

all: $(OUTDIR)/penelope.so

clean:
	$(RM) -r $(OUTDIR)

rebuild: clean all

$(OUTDIR)/penelope.so: blas.c

%.so:
	mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ $^

.PHONY: all clean rebuild watch
