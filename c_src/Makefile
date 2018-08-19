ERLANG_PATH ?= $(shell erl -eval 'io:format("~s", [lists:concat([code:root_dir(), "/erts-", erlang:system_info(version)])])' -s init stop -noshell)
CC           = g++
CFLAGS       = -std=c++11 -Wall -Werror -O3 -fpic \
               -Wl,-undefined,dynamic_lookup -shared \
               -I$(ERLANG_PATH)/include
LIBS         =
OBJDIR       = ../obj
OUTDIR       = ../priv

ifeq ($(shell test -f /usr/lib/libopenblas.so && echo -n yes), yes)
   LIBS += -lopenblas
else
   LIBS += -lblas
endif

all: $(OUTDIR)/penelope.so

clean:
	cd deps && $(MAKE) -f linear.makefile clean
	cd deps && $(MAKE) -f svm.makefile clean
	cd deps && $(MAKE) -f lbfgs.makefile clean
	cd deps && $(MAKE) -f crfsuite.makefile clean
	$(RM) $(OUTDIR)/penelope.so

rebuild: clean all

$(OUTDIR)/penelope.so: init.cpp blas.cpp lin.cpp svm.cpp crf.cpp

%.so:
	mkdir -p $(dir $@)
	cd deps && $(MAKE) -f linear.makefile
	cd deps && $(MAKE) -f svm.makefile
	cd deps && $(MAKE) -f lbfgs.makefile
	cd deps && $(MAKE) -f crfsuite.makefile
	$(CC) $(CFLAGS) -o $@ $^ \
		$(OBJDIR)/liblinear/*.o \
		$(OBJDIR)/libsvm/*.o \
		$(OBJDIR)/liblbfgs/*.o \
		$(OBJDIR)/crfsuite/*.o \
		$(LIBS)

.PHONY: all clean rebuild
