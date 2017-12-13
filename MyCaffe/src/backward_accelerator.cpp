#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>

//#include <ap_axi_sdata.h>
#include "compute.h"
#include "math.h"

using namespace std;

void OutputGradient(ioData expectResult,ioData actualResult,gradientData *output)
{
	u32 r;
	float e;

	if(expectResult.dataSize != actualResult.dataSize)
	{
		cout<<"error param in outputGradient"<<endl;
		return;
	}

	for (r = 0; r < actualResult.dataSize; r++)
    {
		e = actualResult.data[r]-expectResult.data[r];
		output->gradient[r] = e*sigma_derivation(actualResult.data[r]);
	}
}

void Backward(gradientData gradient, weightData weight,  ioData data ,gradientData *output)
{
	u32 r,c;
	float sum = 0;

	if(gradient.gradientSize != weight.outputNum)
	{
		cout<<"error param in backward"<<endl;
		return;
	}
	
   for (c = 0; c < (u32)weight.inputNum; c++)	
   {
   		for (r = 0; r < (u32)(weight.outputNum); r++)
        {
	        sum = gradient.gradient[r]*weight.weight[r][c];
        }
		output->gradient[c] = sum*sigma_derivation(data.data[c]);
		sum = 0;
   }
		
}
