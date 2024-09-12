

FC = gfortran 
LD = gfortran -llapack -lm -lblas #-L/Users/appelo/libs/SuperLU_5.2.1/build/SRC -lsuperlu
LDFLAGS = 
FFLAGS = # -fbounds-check -Wall -fbacktrace -g 
F90FLAGS =  -O3 #-ffpe-summary=none  -fbounds-check -Wall -fbacktrace -g 

OBJECTS = ins.o 
.PHONY: clean


ins.x: $(OBJECTS)
	$(LD) $(OBJECTS) -o ins.x 

%.o : %.f90
	$(FC) $(F90FLAGS) -c $<

%.o : %.f
	$(FC) $(FFLAGS) -c $<

c_fortran_dgssv.o : 
	gcc -c /Users/appelo/libs/SuperLU_5.2.1/FORTRAN/c_fortran_dgssv.c -I/Users/appelo/libs/SuperLU_5.2.1/SRC
clean:
	rm -f $(OBJECTS) ins.x *.txt *.mod
