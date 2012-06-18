CC = g++ -Wall
#CFLAGS = -g -Wall -O3 -ffast-math -DHAVE_INLINE -DGSL_RANGE_CHECK_OFF
# CFLAGS = -g -Wall
LDFLAGS = -lgsl -lm -lgslcblas
                                
LSOURCE =  utils.cpp corpus.cpp state.cpp hdp.cpp main.cpp
LHEADER =  utils.h corpus.h hdp.h state.h

hdp: $(LSOURCE) $(HEADER)
	#$(CC) $(LSOURCE) -o $@ $(LDFLAGS)
	$(CC) $(LSOURCE) -o $@ $(LDFLAGS)

clean:
	rm -f *.o hdp
