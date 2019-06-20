FAST= -fast
CCINTEL=g++
CCMPI=mpicxx
CCMPIINTEL=mpiicpc
MPIFAST=-0fast

FLAGS= -O3 
DEBUG= -g -p -lnlopt
FAST = -O3 -lnlopt
CC = g++
SRCS = opt.cpp  

OBJS = $(SRCS)

main: $(OBJS) 
	$(CCINTEL) -I ~/scratch-mdway/nlopt/include  $(OBJS) -L ~/scratch-midway/nlopt/lib $(FAST) -o run -w -std=c++11 

mpi: $(OBJS)1
	$(CCMPIINTEL) $(OBJS) $(MPIFAST) -lm -w -o dsa

mpi_debug: $(OBJS)
	$(CCMPI)	$(OBJS)	$(DEBUG) -o dsa

intelfast: $(OBJS)
	$(CCINTEL)  $(OBJS) $(FAST) -o dsa

intelstatic: $(OBJS)
	$(CCINTEL)  $(OBJS) $(FAST) -ipo -o dsa

fast: $(OBJS)
	$(CC)  $(OBJS) $(FAST) -o dsa

debug: $(OBJS)
	$(CC)  $(OBJS) $(DEBUG) -o test_run

%.cpp.o: %.cpp
	$(CC) $(FLAGS) $(DEBUG) -c -o $@ $< 
clean:
	rm *.cpp.o

