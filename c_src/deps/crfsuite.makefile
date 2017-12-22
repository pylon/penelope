CC     = gcc
CFLAGS = -c -Wall -O3 -fpic \
         -Iliblbfgs/include -Icrfsuite/include -Icrfsuite/lib/cqdb/include \
         -Wno-unknown-warning-option \
         -Wno-unused-function \
         -Wno-unused-variable \
         -Wno-maybe-uninitialized \
         -Wno-sometimes-uninitialized \
         -Wno-parentheses \
         -Wno-unused-but-set-variable
OBJDIR = ../../obj/crfsuite

all: \
	$(OBJDIR)/cqdb.o \
	$(OBJDIR)/lookup3.o \
	$(OBJDIR)/dictionary.o \
	$(OBJDIR)/logging.o \
	$(OBJDIR)/params.o \
	$(OBJDIR)/quark.o \
	$(OBJDIR)/rumavl.o \
	$(OBJDIR)/dataset.o \
	$(OBJDIR)/holdout.o \
	$(OBJDIR)/train_arow.o \
	$(OBJDIR)/train_averaged_perceptron.o \
	$(OBJDIR)/train_lbfgs.o \
	$(OBJDIR)/train_l2sgd.o \
	$(OBJDIR)/train_passive_aggressive.o \
	$(OBJDIR)/crf1d_context.o \
	$(OBJDIR)/crf1d_model.o \
	$(OBJDIR)/crf1d_feature.o \
	$(OBJDIR)/crf1d_encode.o \
	$(OBJDIR)/crf1d_tag.o \
	$(OBJDIR)/crfsuite_train.o \
	$(OBJDIR)/crfsuite.o

clean: ; $(RM) $(OBJDIR)/*.o

rebuild: clean all

$(OBJDIR)/cqdb.o:                      crfsuite/lib/cqdb/src/cqdb.c
$(OBJDIR)/lookup3.o:                   crfsuite/lib/cqdb/src/lookup3.c
$(OBJDIR)/dictionary.o:                crfsuite/lib/crf/src/dictionary.c
$(OBJDIR)/logging.o:                   crfsuite/lib/crf/src/logging.c
$(OBJDIR)/params.o:                    crfsuite/lib/crf/src/params.c
$(OBJDIR)/quark.o:                     crfsuite/lib/crf/src/quark.c
$(OBJDIR)/rumavl.o:                    crfsuite/lib/crf/src/rumavl.c
$(OBJDIR)/dataset.o:                   crfsuite/lib/crf/src/dataset.c
$(OBJDIR)/holdout.o:                   crfsuite/lib/crf/src/holdout.c
$(OBJDIR)/train_arow.o:                crfsuite/lib/crf/src/train_arow.c
$(OBJDIR)/train_averaged_perceptron.o: crfsuite/lib/crf/src/train_averaged_perceptron.c
$(OBJDIR)/train_lbfgs.o: 					crfsuite/lib/crf/src/train_lbfgs.c
$(OBJDIR)/train_l2sgd.o:               crfsuite/lib/crf/src/train_l2sgd.c
$(OBJDIR)/train_passive_aggressive.o:  crfsuite/lib/crf/src/train_passive_aggressive.c
$(OBJDIR)/crf1d_context.o:             crfsuite/lib/crf/src/crf1d_context.c
$(OBJDIR)/crf1d_model.o:               crfsuite/lib/crf/src/crf1d_model.c
$(OBJDIR)/crf1d_feature.o:             crfsuite/lib/crf/src/crf1d_feature.c
$(OBJDIR)/crf1d_encode.o:              crfsuite/lib/crf/src/crf1d_encode.c
$(OBJDIR)/crf1d_tag.o:                 crfsuite/lib/crf/src/crf1d_tag.c
$(OBJDIR)/crfsuite_train.o:            crfsuite/lib/crf/src/crfsuite_train.c
$(OBJDIR)/crfsuite.o:                  crfsuite/lib/crf/src/crfsuite.c

%.o:
	mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ $^

.PHONY: all clean rebuild
