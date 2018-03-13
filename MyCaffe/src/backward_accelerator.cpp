#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include <fcntl.h>
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
		cout<<"error param in forward"<<endl;
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
		cout<<"error param in forward"<<endl;
		return;
	}
	
   for (c = 0; c < weight.inputNum; c++)	
   {
   		for (r = 0; r < (weight.outputNum); r++)
        {
	        sum = sum+gradient.gradient[r]*weight.weight[r][c];
        }
		output->gradient[c] = sum*sigma_derivation(data.data[c]);
		sum = 0;
   }
		
}

void Backward_fullconnect_afterpool(gradientData gradient, weightData weight, gradientData *output)
{
	u32 r,c;
	float sum = 0;

	if(gradient.gradientSize != weight.outputNum)
	{
		cout<<"error param in forward"<<endl;
		return;
	}
	
   for (c = 0; c < weight.inputNum; c++)	
   {
   		for (r = 0; r < (weight.outputNum); r++)
        {
	        sum = sum+gradient.gradient[r]*weight.weight[r][c];
        }
		output->gradient[c] = sum;
		sum = 0;
   }
		
}

void Backward_fullconnect_afterpool_FPGA(gradientData gradient, weightData weight, gradientData *output, int fd, int fd11)
{
	u32 r,c;
	float sum = 0;
    int n,type=2;

    write(fd11, &n, sizeof(n));         //init the ip core
    
    write(fd, &type, sizeof(type));
    write(fd, &(weight.outputNum), sizeof(weight.outputNum));
    write(fd, &(weight.inputNum), sizeof(weight.inputNum));

    write(fd, gradient.gradient, gradient.gradientSize*sizeof(float));
	for (int i = 0; i < weight.outputNum; i++)
    		write(fd, weight.weight[i], weight.inputNum*sizeof(float));
    read(fd, output->gradient, output->gradientSize*sizeof(float));
}


void Backward_conv(int stride, convData gradient, convWeight weight, convData *output)
{
	u32 i,j,r,c;
	float sum = 0;

    float **temp = gettempbuffer();
    float **outputTemp = getoutputtempbufer();
    float **flipmap = getfliptempbuffer();

	nSize outSize = {output->height,output->height}; 
    nSize inSize = {gradient.height, gradient.height}; 
    nSize mapSize = {weight.kernelSize, weight.kernelSize};
	for (i = 0; i < weight.inputNum; i++) {
		for(j = 0; j < weight.outputNum; j++) {
            init_malloc_buffer();
			rotate180(weight.weight[j][i], mapSize,(float **)flipmap); //旋转180度的特征模板
			convolution((float **)flipmap, mapSize, gradient.img_data[j], inSize, full,(float **)temp,(float **)outputTemp);
			add_mat(output->img_data[i], output->img_data[i], outSize, (float **)outputTemp, outSize);
		}
	}
}
void Backward_conv_FPGA(int stride, convData gradient, convWeight weight, convData *output, int fd, int fd11)
{
    int n,type=0;

    write(fd11, &n, sizeof(n));         //init the ip core
    
    write(fd, &type, sizeof(type));
    write(fd, &(weight.outputNum), sizeof(weight.outputNum));
    write(fd, &(weight.inputNum), sizeof(weight.inputNum));
    write(fd, &(gradient.height), sizeof(gradient.height));
    write(fd, &(output->height), sizeof(output->height));
    write(fd, &(weight.kernelSize), sizeof(weight.kernelSize));
    
    write(fd, **(gradient.img_data), gradient.height*gradient.height*gradient.img_num*sizeof(float));
	for (int i=0; i < weight.outputNum;i++)
		for(int j=0; j< weight.inputNum;j++)
			for(int k=0;k<weight.kernelSize;k++)
    write(fd, weight.weight[i][j][k], weight.kernelSize*sizeof(float));
    read(fd, **(output->img_data), output->height*output->height*output->img_num*sizeof(float));
}

void Backward_pool(int kernelSize,int stride, convData gradient, convData data,convData *output)
{
	u32 i,j,r,c;
	for (i = 0; i < output->img_num; i++) {
		for (r = 0; r < gradient.height*kernelSize; r++)
			for (c = 0; c < gradient.height*kernelSize; c++)
				output->img_data[i][r][c] = gradient.img_data[i][r/kernelSize][c/kernelSize] * sigma_derivation(data.img_data[i][r][c]) / ((float)(kernelSize * kernelSize));
    }
}

void Backward_pool_FPGA(int kernelSize,int stride, convData gradient, convData data,convData *output, int fd, int fd11)
{
    int n,type=1;

    write(fd11, &n, sizeof(n));         //init the ip core
    
    write(fd, &type, sizeof(type));
    write(fd, &(gradient.img_num), sizeof(gradient.img_num));
    write(fd, &(data.img_num), sizeof(data.img_num));
    write(fd, &(gradient.height), sizeof(gradient.height));
    write(fd, &(output->height), sizeof(output->height));
    write(fd, &(kernelSize), sizeof(kernelSize));
    
	for(int i=0;i < gradient.img_num;i++)
		for (int j=0;j< gradient.height;j++)
    			write(fd, gradient.img_data[i][j], gradient.height*sizeof(float));
	for(int i=0;i < data.img_num;i++)
		for (int j=0;j< data.height;j++)
    write(fd, data.img_data[i][j], data.height*sizeof(float));
    
    read(fd, **(output->img_data), output->height*output->height*output->img_num*sizeof(float));

	printf("read done.\n");
}
