#include "math.h"
#include <iostream>
#include <stdio.h>

#ifdef USE_MALLOC_TEMP
static float **temp_buffer;
static float **flipTemp_buffer;
static float **outputTemp_buffer;
#else
static float **temp_buffer;
static float **flipTemp_buffer;
static float **outputTemp_buffer;
static float* temp_buffer_p[SUPPORT_MAX];
static float* outputTemp_buffer_p[SUPPORT_MAX];
static float* flipTemp_buffer_p[KERNEL_MAX];
static float temp_buffer_arr[SUPPORT_MAX][SUPPORT_MAX];
static float outputTemp_buffer_arr[SUPPORT_MAX][SUPPORT_MAX];
static float flipTemp_buffer_arr[KERNEL_MAX][KERNEL_MAX];
#endif

// 激励函数sigmoid
float activation_Sigma(float input,float bas)
{
        float temp = input + bas;
        return (float)1.0 / ((float)(1.0 + exp(-temp)));
}

float sigma_derivation(float y) // Logic
{ 
        return y * (1 - y); // y是经过激活函数的输出值
}

/*
	full指完全，操作后结果的大小为inSize+(mapSize-1)
	same指同输入相同大小
	valid指完全操作后的大小，一般为inSize-(mapSize-1)大小，其不需要将输入添0扩大
*/
void convolution(float** map, nSize mapSize, float** inputData, nSize inSize, int type, float** temp,float** outputData)// 互相关
{
	int i, j, c, r;

	int outSizeW = inSize.c - (mapSize.c - 1); // 默认前向
	int outSizeH = inSize.r - (mapSize.r - 1);
	int addr = mapSize.r - 1;
	int addc = mapSize.c - 1;
	float** exInputData = inputData;
	if(type == full)
		for (j = 0; j < inSize.r + 2 * addr; j++) {
		for (i = 0; i < inSize.c + 2 * addc; i++) {
			if (j < addr || i < addc || j >= (inSize.r + addr) || i >= (inSize.c + addc))
				temp[j][i] = (float)0.0;
			else
				temp[j][i] = inputData[j-addr][i-addc]; // 复制原向量的数据
		}
		exInputData = temp;
		outSizeW = inSize.c + (mapSize.c - 1); 
		outSizeH = inSize.r + (mapSize.r - 1);
	}
	for(j = 0; j < outSizeH; j++)
		for (i = 0; i < outSizeW; i++)
			for (r = 0; r < mapSize.r; r++)
				for (c = 0; c < mapSize.c; c++) {
					outputData[j][i] = outputData[j][i] + map[r][c] * exInputData[j+r][i+c];
				}
    return;
}


void rotate180(float** mat, nSize matSize,float** outputData)// 矩阵翻转180度
{
	int i, c, r;
	int outSizeW = matSize.c;
	int outSizeH = matSize.r;
	
	for (r = 0; r < outSizeH; r++)
		for (c = 0; c < outSizeW; c++)
			outputData[r][c] = mat[outSizeH-r-1][outSizeW-c-1];

	return;
}


void add_mat(float** res, float** mat1, nSize matSize1, float** mat2, nSize matSize2)// 矩阵相加
{
	int i, j;
	if (matSize1.c != matSize2.c || matSize1.r != matSize2.r)
        std::cout<<"ERROR: Size is not same!"<<std::endl;

	for (i = 0; i < matSize1.r; i++)
		for (j = 0; j < matSize1.c; j++)
			res[i][j] = mat1[i][j] + mat2[i][j];
}

void multi_factor(float** res, float** mat, nSize matSize, float factor)// 矩阵乘以系数
{
	int i, j;
	for (i = 0; i < matSize.r; i++)
		for (j = 0; j < matSize.c; j++)
			res[i][j] = mat[i][j] * factor;
}

float sum_mat(float** mat, nSize matSize) // 矩阵各元素的和
{
	float sum = 0.0;
	int i, j;
	for (i = 0; i < matSize.r; i++)
		for (j = 0; j < matSize.c; j++)
			sum = sum + mat[i][j];
	return sum;
}

void malloc_temp_buffer(){
#ifdef USE_MALLOC_TEMP
	int i;
	temp_buffer = new float *[SUPPORT_MAX];
    outputTemp_buffer = new float *[SUPPORT_MAX];
    flipTemp_buffer = new float *[KERNEL_MAX];
    for(i = 0;i<SUPPORT_MAX;i++){
        temp_buffer[i] = new float [SUPPORT_MAX];
        outputTemp_buffer[i] = new float [SUPPORT_MAX];
    }
    for(i = 0;i<KERNEL_MAX;i++)
    flipTemp_buffer[i] = new float [KERNEL_MAX];
#else
	int i,j;
	for(i = 0;i<SUPPORT_MAX;i++){
		temp_buffer_p[i] = temp_buffer_arr[i];
		outputTemp_buffer_p[i] = outputTemp_buffer_arr[i];
	}
	for(i = 0;i<KERNEL_MAX;i++)
		flipTemp_buffer_p[i] = flipTemp_buffer_arr[i];

	temp_buffer = temp_buffer_p;
	flipTemp_buffer = flipTemp_buffer_p;
	outputTemp_buffer = outputTemp_buffer_p;
#endif
}

float **gettempbuffer(){
	return (float **)temp_buffer;
}
float **getoutputtempbufer(){
	return (float **)outputTemp_buffer;
}
float **getfliptempbuffer(){
	return (float **)flipTemp_buffer;
}

void init_malloc_buffer(){
    int i,j;
    for(i = 0;i<SUPPORT_MAX;i++){
        for(j = 0;j<SUPPORT_MAX;j++){
            temp_buffer[i][j] = 0.0;
            outputTemp_buffer[i][j] = 0.0;
        }
    }
    for(i = 0;i<KERNEL_MAX;i++)
        for(j = 0;j<KERNEL_MAX;j++)
    flipTemp_buffer[i][j] = 0.0;

}

void print_mat(char* name,float **mat,int row,int col){
    printf("%s\n",name);
    for(int r = 0 ;r<row;r++){
        for(int c = 0;c<col;c++)
            printf("%.1f ",mat[r][c]);
        printf("\n");
    }
    
}

void print_vector(char* name,float *vector,int col){
    printf("%s\n",name);
    for(int c = 0;c<col;c++)
        printf("%.1f ",vector[c]);
    printf("\n");
}
