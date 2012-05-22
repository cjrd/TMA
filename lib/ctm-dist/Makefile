.SUFFIXES: .c .u
CC= gcc

CFLAGS = -g -Wall -O3 -DHAVE_INLINE -DGSL_RANGE_CHECK_OFF
LDFLAGS = -lm -lgsl -lgslcblas
# LDFLAGS = -lm -lgsl -latlas -lcblas
LOBJECTS= inference.o gsl-wrappers.o ctm.o estimate.o corpus.o params.o
LSOURCE= inference.c gsl-wrappers.c corpus.c estimate.c corpus.c params.c

mac:	$(LOBJECTS)
	$(CC) $(LOBJECTS) -o ctm $(LDFLAGS)

linux:	$(LOBJECTS)
	$(CC) $(LOBJECTS) -o ctm $(LDFLAGS)

debug:	$(LOBJECTS)
	$(CC) $(LOBJECTS) -o ctm $(LDFLAGS)

clean:
	-rm -f *.o
