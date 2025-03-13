# makefile
#include ../../flags.mk

LOBJ = ./objects
FFLAGS = -O1 -fno-automatic

OBJS = $(LOBJ)/rayleigh_python.o $(LOBJ)/ltable_python.o
SRCS = rayleigh_python.f ltable_python.f
F2PY = f2py
F77 = gfortran


all: rayleigh_python.so

rayleigh_python.so: $(OBJS) rayleigh_python.pyf
	$(F2PY) -c rayleigh_python.pyf $(OBJS)
	mv rayleigh_python.*.so rayleigh_python.so

rayleigh_python.pyf: $(SRCS)
	$(F2PY) --overwrite-signature -m rayleigh_python -h rayleigh_python.pyf $(SRCS)

clean:
	$(RM) $(LOBJ)/*.o
very_clean: clean
	$(RM) rayleigh_python.pyf rayleigh_python.so 

$(LOBJ)/%.o: %.f
	$(F77) $(FFLAGS) -c $*.f -o $(LOBJ)/$*.o
#$(LOBJ)/%.o: %.f90
#	$(F90) $(F90FLAGS) -c $*.f90 -o $(LOBJ)/$*.o

