#include <stdio.h>
#include <math.h>

double log_gamma(double x)
{
	double z=1/(x*x);

	x=x+6;
	z=(((-0.000595238095238*z+0.000793650793651)
		*z-0.002777777777778)*z+0.083333333333333)/x;
	z=(x-0.5)*log(x)-x+0.918938533204673+z-log(x-1)-
		log(x-2)-log(x-3)-log(x-4)-log(x-5)-log(x-6);
	return z;
}

int main(int argc, char* argv[]){
	printf("lgamma is %0.5f log_gamma is %0.5f \n",lgamma(atof(argv[1])),log_gamma(atof(argv[1])));
}

