#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>

//#include <ap_axi_sdata.h>
#include "compute.h"
#include "math.h"

using namespace std;

void Forward(ioData input, weightData weight,ioData *output)
{
	u32 r,c;
	float sum = 0;

	if(input.dataSize != weight.inputNum)
	{
		cout<<"error param in forward"<<endl;
		return;
	}
    for (r = 0; r < (weight.outputNum); r++)
    {
		sum = 0;
		for(c = 0; c < (weight.inputNum); c++)
		{
			sum = input.data[c]*weight.weight[r][c];
		}
		output->data[r] = activation_Sigma(sum, weight.weight[r][weight.inputNum]);
    }
}
