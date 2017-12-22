CC     = g++
CFLAGS = -c -Wall -O3 -fpic
OBJDIR = ../../obj/liblinear

all: $(OBJDIR)/linear.o $(OBJDIR)/tron.o

clean: ; $(RM) $(OBJDIR)/*.o

rebuild: clean all

$(OBJDIR)/linear.o: liblinear/linear.cpp
$(OBJDIR)/tron.o: liblinear/tron.cpp

%.o:
	mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ $^

.PHONY: all clean rebuild
