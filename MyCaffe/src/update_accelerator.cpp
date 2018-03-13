#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>

//#include <ap_axi_sdata.h>
#include "compute.h"
#include "math.h"

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
    for (r = 0; r < (gradient.gradientSize); r++)
	{
        for (c = 0; c < (data.dataSize); c++)
		{
			temp = gradient.gradient[r]*data.data[c];
			output->weight[r][c] = output->weight[r][c]-TRAIN_RATE*temp;
		}       
        output->weight[r][c] = output->weight[r][c]-TRAIN_RATE*gradient.gradient[r];
    }
}

void ApplyUpdata_conv(int stride, convData gradient, convData data, convWeight *output)
{
	u32 i,j,r,c;

    float **temp = gettempbuffer();
    float **outputTemp = getoutputtempbufer();
    float **flipmap = getfliptempbuffer();

    nSize dSize = {gradient.height, gradient.height};
    nSize ySize = {data.height, data.height};
    nSize mapSize = {output->kernelSize, output->kernelSize};

    for (i = 0; i < gradient.img_num; i++) { 
        for (j = 0; j < data.img_num; j++) {
            init_malloc_buffer();
            convolution(gradient.img_data[i], dSize, data.img_data[j], ySize, valid,(float **)temp,(float **)outputTemp);
            multi_factor((float **)outputTemp, (float **)outputTemp, mapSize, -1 * TRAIN_RATE); // 矩阵乘以系数(学习率)
            add_mat(output->weight[i][j], output->weight[i][j], mapSize, (float **)outputTemp, mapSize);
              
        }
        output->bias.data[i] -= TRAIN_RATE * sum_mat(gradient.img_data[i], dSize); //summat求矩阵格元素的和
    }
}
