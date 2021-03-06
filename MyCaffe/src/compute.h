#ifndef COMPUTE_H
#define COMPUTE_H

#include "math.h"

typedef unsigned int u32;

typedef struct IOData       //inputdata and outputdata
{
    float *data;
    int dataSize;
}ioData;

typedef struct GradientData     //gradient data
{
    float *gradient;
    int gradientSize;
}gradientData;

typedef struct WeightData       //weightdata
{
    float **weight;
    int inputNum;
    int outputNum;
}weightData;

typedef struct ConvData
{
    int height; //输入图片的宽高，正方形图片
    int img_num; //输入图片的数量
    float ***img_data;
}convData;

typedef struct ConvWeight
{
    int inputNum;
    int outputNum;
    int kernelSize;
    float ****weight;
	IOData bias;
}convWeight;


//暂时使用用固定学习率，如果使用动态学习率需要与权重对应输入
#define TRAIN_RATE 1.1
//#define USE_MALLOC_TEMP

void OutputGradient(ioData expectResult,ioData actualResult,gradientData *output);

/*
第l层网络，input为l层的输入，weight为l层的权重，大小为inputSize*outputSize
返回ouput 为下一层的输入

☆☆特注1：bias看做固定输入为1的权重，放在weight的最后一个，一层有outpuSize个bias☆☆
☆☆特注2：权重矩阵行为输出 列为输入的顺序放，即weight[outsize][inputsize]☆☆
*/
void Forward(ioData input, weightData weight,ioData *output);

/*
第l层网络，input为l+1层的梯度，weight为l+1层的权重，data为l层的输出结果矩阵
返回ouput为l层的梯度
*/
void Backward(gradientData gradient, weightData weight,  ioData data ,gradientData *output);

void Backward_fullconnect_afterpool(gradientData gradient, weightData weight, gradientData *output);

void Backward_fullconnect_afterpool_FPGA(gradientData gradient, weightData weight, gradientData *output, int fd, int fd11);
/*
第l层网络，gradient为l层的梯度，data为l层的输入
返回output为修改后l层的权重
*/
void ApplyUpdate(gradientData gradient, ioData data,weightData *output);


void Forward_conv(int stride, convData input, convWeight weight, convData *output);

void Backward_conv(int stride, convData gradient, convWeight weight, convData *output);

void Backward_conv_FPGA(int stride, convData gradient, convWeight weight, convData *output, int fd, int fd11);

void Forward_pool(convData input, int kernelSize, int stride, convData *output);

void Backward_pool(int kernelSize,int stride, convData gradient, convData data,convData *output);

void Backward_pool_FPGA(int kernelSize,int stride, convData gradient, convData data,convData *output, int fd, int fd11);
void ApplyUpdata_conv(int stride, convData gradient, convData data, convWeight *output);

#endif
