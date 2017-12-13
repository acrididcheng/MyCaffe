#include "math.h"

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


