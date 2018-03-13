#ifndef MATH_H
#define MATH_H

#include <math.h>

#define bitwidth 32
#define SUPPORT_MAX 32 //定义支持最大矩阵
#define KERNEL_MAX 5
#define full 0
#define same 1
#define valid 2
typedef unsigned int u32;

// 表示矩阵长宽的结构
typedef struct Mat2DSize {
        int c; // 列
        int r; // 行
} nSize;

float activation_Sigma(float input,float bas);
float sigma_derivation(float y);
void rotate180(float** mat, nSize matSize,float** outputData);
void convolution(float** map, nSize mapSize, float** inputData, nSize inSize, int type, float** temp,float** outputData);
void multi_factor(float** res, float** mat, nSize matSize, float factor);
void add_mat(float** res, float** mat1, nSize matSize1, float** mat2, nSize matSize2);
float sum_mat(float** mat, nSize matSize);

float **gettempbuffer();
float **getoutputtempbufer();
float **getfliptempbuffer();
void malloc_temp_buffer();
void print_mat(char* name,float **mat,int row,int col);
void print_vector(char* name,float *vector,int col);
void init_malloc_buffer();
#endif
