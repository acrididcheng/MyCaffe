#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>

//#include <ap_axi_sdata.h>
#include "compute.h"

using namespace std;

void ApplyUpdate(gradientData gradient, ioData data,weightData *output)
{
	u32 r,c;
	float temp;
	
	if((gradient.gradientSize!=output->outputNum) || (data.dataSize!=output->inputNum))
	{
		cout<<"error param in forward"<<endl;
		return;
	}
    for (r = 0; r < (u32)(gradient.gradientSize); r++) {
            for (c = 0; c < (u32)(data.dataSize); c++)
			{
				temp = gradient.gradient[r]*data.data[c];
				output->weight[r][c] = output->weight[r][c]-TRAIN_RATE*temp;
			}       
            output->weight[r][c] = output->weight[r][c]-TRAIN_RATE*gradient.gradient[r];
    }
}
