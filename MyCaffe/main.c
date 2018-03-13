#include ".\\include\\common.h"
#include ".\\include\\first_convolution.h"

static input_type input_images[32][32] = {0};
static NET_STATE cur_state = TRAINING;
//读取进来的28*28图片一维数组转换成四周填0 的32*32二维数组
void transform_image(u8 image_buffer[784])
{
    u8 i,j;
    for(i = 0;i<=27;i++)
        for(j = 0;j<=27;j++)
            input_images[i+2][j+2]=(input_type)image_buffer[i*28+j]/255.0;
}

void init_layers()
{
    init_first_convolution();
    init_first_subSampling();
    init_second_convolution();
    init_second_subSampling();
    init_third_convolution();
    init_full_connect();
    init_final_output();
    cur_state = TRAINING;
}

NET_STATE get_net_state()
{
    return cur_state;
}
int main()
{
    FILE *fp;
    u8 buffer[16]={0};
    u8 image_buffer[784]={0};
    u8 i;
    u32 count = 0;
    if ((fp=fopen(".\\images\\train-images.idx3-ubyte","rb"))==NULL){
        printf("打开文件错误\n");
        return NULL;
    }
    fread(buffer,1,sizeof(buffer),fp);
    for(i=0;i<=15;i++)
    {
        printf("%x ",buffer[i]);
    }
    printf("\n");
    init_layers();
    //目前先读取一张图片，后期需要加while循环

    while(fread(image_buffer,1,sizeof(image_buffer),fp)>0)
    {
        {
            u8 i,j;
            for(i = 0;i<=27;i++){
                for(j = 0;j<=27;j++)
                    printf("%3d ",image_buffer[i*28+j]);
                printf("\n");
            }
        }
        transform_image(image_buffer);
        //进入第一层卷积开始正向传播
        start_first_convolution(input_images);
        printf("正在处理第%d 张图片\n ",++count);
        //if(i++==50)
           // break;
    }
    final_output_terminal();
    printf("image training has done\n");
    getchar();
    fclose(fp);

    //开始测试训练结果
    if ((fp=fopen(".\\images\\t10k-images.idx3-ubyte","rb"))==NULL){
        printf("打开文件错误\n");
        return NULL;
    }
    fread(buffer,1,sizeof(buffer),fp);
    for(i=0;i<=15;i++)
    {
        printf("%x ",buffer[i]);
    }
    final_output_open_test_label();
    cur_state = TEST;
    while(fread(image_buffer,1,sizeof(image_buffer),fp)>0)
    {
        {
            u8 i,j;
            for(i = 0;i<=27;i++){
                for(j = 0;j<=27;j++)
                    printf("%3d ",image_buffer[i*28+j]);
                printf("\n");
            }
        }
        transform_image(image_buffer);
        //进入第一层卷积开始正向传播
        start_first_convolution(input_images);
        printf("正在测试第%d 张图片\n ",++count);

    }

    printf("正确率是：%f",get_test_result());
    final_output_terminal();
    return 0;
}
