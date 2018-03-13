#include <fcntl.h>
#include <unistd.h>

#include <iostream>
#include <string>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <glog/logging.h>
#include <gflags/gflags.h>
#include <stdio.h>

#include "caffe.pb.h"
#include "parambuf.hpp"
#include "mynet.hpp"
#include "compute.h"
using namespace caffe;
using namespace std;
#define INPUT_NUM 5
#define OUTPUT_NUM 10
#define DATA_NUM 1
typedef unsigned char u8;

float inputmatrix[DATA_NUM][INPUT_NUM]=
{
    {0.1,0.5,0.8,0.9,1.0},
    //{0.2,0.4,0.1,0.5,0.4},
    //{0.2,0.1,0.5,0.3,0.5}
};

float outputmatrix[DATA_NUM][OUTPUT_NUM]=
{
    {0.1,0.5,0.8,0.9,1.0},
    //{0.2,0.4,0.1,0.5,0.4},
    //{0.2,0.1,0.5,0.3,0.5}
};


using google::protobuf::io::FileInputStream;
using google::protobuf::Message;
bool ReadProtoFromTextFile(const char* filename, Message* proto) {
  int fd = open(filename, O_RDONLY);
  FileInputStream* input = new FileInputStream(fd);
  bool success = google::protobuf::TextFormat::Parse(input, proto);
  delete input;
  close(fd);
  return success;
}

void GlobalInit( MyNet *mynet)
{
    int i;
    for (i = 0; i < mynet->getLayerNum(); i++)
    {
        mynet->getML(i)->GradientInit();
        mynet->getML(i)->WeightInit();
        mynet->getML(i)->OutdataInit();
    }
}

void Step( MyNet *mynet, float ***input, float *output, int isFPGA, int fd, int fd11)
{
    int i;

    //Forward procedure
    ioData tempinputData,tempoutputData,tempData;
    weightData tempweightData;
    gradientData tempgradientData,tempgradientDataOut;

	convData tempconvinputData,tempconvoutputData,tempconvgradientData,tempconvgradientDataOut;
	convWeight tempconvweightData;
	
    tempconvweightData.inputNum = mynet->getML(0)->getInputnum();
    for (i = 0; i < mynet->getLayerNum(); i++)
    {
    	if(mynet->getML(i)->getLayerType() == CONVOL){
			tempconvinputData.height = mynet->getML(i)->getInputImageSize();
		    tempconvinputData.img_num = mynet->getML(i)->getInputnum();
			if(i == 0)
				tempconvinputData.img_data = input;
			else
				tempconvinputData.img_data = mynet->getML(i-1)->getConvOutdata();
			
			tempconvoutputData.height = mynet->getML(i)->getOutputImageSize();
			tempconvoutputData.img_num = mynet->getML(i)->getOutputnum();
			tempconvoutputData.img_data = mynet->getML(i)->getConvOutdata();
			
			tempconvweightData.inputNum = mynet->getML(i)->getInputnum();
			tempconvweightData.outputNum = mynet->getML(i)->getOutputnum();
			tempconvweightData.kernelSize = mynet->getML(i)->getKernelSize();
			tempconvweightData.bias.dataSize = mynet->getML(i)->getOutputnum();
			tempconvweightData.bias.data = mynet->getML(i)->getConvBias();
			tempconvweightData.weight = mynet->getML(i)->getConvWeight();
            Forward_conv(1, tempconvinputData, tempconvweightData, &tempconvoutputData);
           //print_mat("after output image",tempconvoutputData.img_data[0],tempconvoutputData.height,tempconvoutputData.height);
		}
        else if(mynet->getML(i)->getLayerType() == POOL){
			tempconvinputData.height = mynet->getML(i)->getInputImageSize();
		    tempconvinputData.img_num = mynet->getML(i)->getInputnum();
			tempconvinputData.img_data = mynet->getML(i-1)->getConvOutdata();

			tempconvoutputData.height = mynet->getML(i)->getOutputImageSize();
			tempconvoutputData.img_num = mynet->getML(i)->getOutputnum();
			tempconvoutputData.img_data = mynet->getML(i)->getConvOutdata();
			Forward_pool(tempconvinputData, 2, 1, &tempconvoutputData);
            //print_mat("after",tempconvoutputData.img_data[0],tempconvoutputData.height,tempconvoutputData.height);
		}
        else if(mynet->getML(i)->getLayerType() == FULLCON){
            tempinputData.dataSize = mynet->getML(i)->getInputnum();
            tempconvinputData.img_data = mynet->getML(i-1)->getConvOutdata();
			tempinputData.data = new float[tempinputData.dataSize];
			int outsize = mynet->getML(i-1)->getOutputImageSize();
			for (int k = 0; k < (mynet->getML(i-1)->getOutputnum()); k++)
				for (int r = 0; r < outsize; r++)
					for (int c = 0; c < outsize; c++)
						tempinputData.data[k * outsize * outsize + r * outsize + c] = tempconvinputData.img_data[k][r][c];
		
			tempweightData.inputNum = mynet->getML(i)->getInputnum();
	        tempweightData.outputNum = mynet->getML(i)->getOutputnum();	        
	        tempweightData.weight = mynet->getML(i)->getWeight();

	        tempoutputData.dataSize = mynet->getML(i)->getOutputnum();
	        tempoutputData.data = mynet->getML(i)->getOutdata();
	        Forward(tempinputData, tempweightData, &tempoutputData);
            //print_vector("after forward",tempoutputData.data,tempoutputData.dataSize);
			//delete(tempinputData.data);
		}
    }
    tempinputData.dataSize = mynet->getML(mynet->getLayerNum()-1)->getOutputnum();
    tempinputData.data = output;

    tempData.dataSize = mynet->getML(mynet->getLayerNum()-1)->getOutputnum();
    tempData.data = mynet->getML(mynet->getLayerNum()-1)->getOutdata();
	for(int k = 0;k<tempData.dataSize;k++)
		printf("The possibility of the number %d is: %f  \n",k,tempData.data[k]);
    
    tempgradientData.gradientSize = mynet->getML(mynet->getLayerNum()-1)->getOutputnum();
    tempgradientData.gradient = mynet->getML(mynet->getLayerNum()-1)->getGradient();

    OutputGradient( tempinputData, tempData, &tempgradientData );
    //print_vector("after output gradient",tempgradientData.gradient,tempgradientData.gradientSize);
    for (i = mynet->getLayerNum() - 1; i > 0; i--)
    {
    	if(mynet->getML(i)->getLayerType() == CONVOL){
			tempconvgradientData.height = mynet->getML(i)->getOutputImageSize();
			tempconvgradientData.img_num = mynet->getML(i)->getOutputnum();
			tempconvgradientData.img_data = mynet->getML(i)->getConvGradient();

			tempconvweightData.inputNum = mynet->getML(i)->getInputnum();
			tempconvweightData.outputNum = mynet->getML(i)->getOutputnum();
			tempconvweightData.bias.data = mynet->getML(i)->getConvBias();
			tempconvweightData.bias.dataSize = mynet->getML(i)->getOutputnum();
			tempconvweightData.kernelSize = mynet->getML(i)->getKernelSize();
			tempconvweightData.weight = mynet->getML(i)->getConvWeight();

			tempconvgradientDataOut.height = mynet->getML(i-1)->getOutputImageSize();
			tempconvgradientDataOut.img_num = mynet->getML(i-1)->getOutputnum();
			tempconvgradientDataOut.img_data = mynet->getML(i-1)->getConvGradient();
			if (isFPGA == 1)
                Backward_conv_FPGA(1, tempconvgradientData, tempconvweightData, &tempconvgradientDataOut,fd,fd11);
            else
                Backward_conv(1, tempconvgradientData, tempconvweightData, &tempconvgradientDataOut);
           //print_mat("after conv output gradient",tempconvgradientData.img_data[0],tempconvgradientData.height,tempconvgradientData.height);
		}
		else if(mynet->getML(i)->getLayerType() == POOL){
			tempconvgradientData.height = mynet->getML(i)->getOutputImageSize();
			tempconvgradientData.img_num = mynet->getML(i)->getOutputnum();
			tempconvgradientData.img_data = mynet->getML(i)->getConvGradient();

			tempconvgradientDataOut.height = mynet->getML(i-1)->getOutputImageSize();
			tempconvgradientDataOut.img_num = mynet->getML(i-1)->getOutputnum();
			tempconvgradientDataOut.img_data = mynet->getML(i-1)->getConvGradient();

			tempconvoutputData.height = mynet->getML(i-1)->getOutputImageSize();
			tempconvoutputData.img_num = mynet->getML(i-1)->getOutputnum();
			tempconvoutputData.img_data = mynet->getML(i-1)->getConvOutdata();
			if (isFPGA == 1)
                Backward_pool_FPGA(2,1, tempconvgradientData, tempconvoutputData,&tempconvgradientDataOut,fd, fd11);
            else 
                Backward_pool(2,1, tempconvgradientData, tempconvoutputData,&tempconvgradientDataOut);
            //print_mat("after pool gradient",tempconvgradientData.img_data[0],tempconvgradientData.height,tempconvgradientData.height);
		}
		else if(mynet->getML(i)->getLayerType() == FULLCON){
			tempgradientData.gradientSize = mynet->getML(i)->getOutputnum();
	        tempgradientData.gradient = mynet->getML(i)->getGradient();

	        tempweightData.inputNum = mynet->getML(i)->getInputnum();
	        tempweightData.outputNum = mynet->getML(i)->getOutputnum();
	        tempweightData.weight = mynet->getML(i)->getWeight();
	    
	        tempgradientDataOut.gradientSize = mynet->getML(i)->getInputnum();
	        tempgradientDataOut.gradient = new float[mynet->getML(i)->getInputnum()];
	        if (isFPGA == 1)
                Backward_fullconnect_afterpool_FPGA( tempgradientData, tempweightData, &tempgradientDataOut,fd , fd11);
            else
                Backward_fullconnect_afterpool( tempgradientData, tempweightData, &tempgradientDataOut);
			int outsize = mynet->getML(i-1)->getOutputImageSize();
			tempconvgradientData.img_data = mynet->getML(i-1)->getConvGradient();
			for (int k = 0; k< mynet->getML(i-1)->getOutputnum(); k++)
                for (int r = 0; r < outsize; r++)
                	for (int c = 0; c < outsize; c++)
                       tempconvgradientData.img_data[k][r][c] = tempgradientDataOut.gradient[k * outsize * outsize + r * outsize + c];
			//print_vector("after full gradient",tempgradientDataOut.gradient,tempgradientDataOut.gradientSize);		
			//delete(tempgradientDataOut.gradientconvweight[i][j][r]);
		}
        
    }
    //return;
    for (i = 0; i < mynet->getLayerNum(); i++)
    {
    	if(mynet->getML(i)->getLayerType() == CONVOL){
			tempconvgradientData.height = mynet->getML(i)->getOutputImageSize();
			tempconvgradientData.img_num = mynet->getML(i)->getOutputnum();
			tempconvgradientData.img_data = mynet->getML(i)->getConvGradient();

			tempconvinputData.height = mynet->getML(i)->getInputImageSize();
			tempconvinputData.img_num = mynet->getML(i)->getInputnum();
			if(i == 0)
				tempconvinputData.img_data = input;
			else
				tempconvinputData.img_data = mynet->getML(i-1)->getConvOutdata();
			
			tempconvweightData.inputNum = mynet->getML(i)->getInputnum();
			tempconvweightData.outputNum = mynet->getML(i)->getOutputnum();
			tempconvweightData.bias.data = mynet->getML(i)->getConvBias();
			tempconvweightData.bias.dataSize = mynet->getML(i)->getOutputnum();
			tempconvweightData.kernelSize = mynet->getML(i)->getKernelSize();
			tempconvweightData.weight = mynet->getML(i)->getConvWeight();
			ApplyUpdata_conv(1, tempconvgradientData, tempconvinputData, &tempconvweightData);
		}
		else if(mynet->getML(i)->getLayerType() == POOL){
			continue;
		}
		else if(mynet->getML(i)->getLayerType() == FULLCON){
			tempgradientData.gradientSize = mynet->getML(i)->getOutputnum();
	        tempgradientData.gradient = mynet->getML(i)->getGradient();
	        
	        tempData.dataSize = mynet->getML(i)->getInputnum();
            tempconvinputData.img_data = mynet->getML(i-1)->getConvOutdata();
			tempData.data = new float[tempData.dataSize];
			int outsize = mynet->getML(i-1)->getOutputImageSize();
			for (int k = 0; k < (mynet->getML(i-1)->getOutputnum()); k++)
				for (int r = 0; r < outsize; r++)
					for (int c = 0; c < outsize; c++)
						tempData.data[k * outsize * outsize + r * outsize + c] = tempconvinputData.img_data[k][r][c];

	        tempweightData.inputNum = mynet->getML(i)->getInputnum();
	        tempweightData.outputNum = mynet->getML(i)->getOutputnum();
	        tempweightData.weight = mynet->getML(i)->getWeight();
	        
	        ApplyUpdate(tempgradientData, tempData, &tempweightData);
			//delete(tempData.data);
		}
    }
}

int mytest(MyNet *mynet, float ***input, int result){
	int i;
	
	//Forward procedure
	ioData tempinputData,tempoutputData,tempData;
	weightData tempweightData;
	gradientData tempgradientData,tempgradientDataOut;

	convData tempconvinputData,tempconvoutputData,tempconvgradientData,tempconvgradientDataOut;
	convWeight tempconvweightData;
	
	tempconvweightData.inputNum = mynet->getML(0)->getInputnum();
	for (i = 0; i < mynet->getLayerNum(); i++)
	{
		if(mynet->getML(i)->getLayerType() == CONVOL){
			tempconvinputData.height = mynet->getML(i)->getInputImageSize();
			tempconvinputData.img_num = mynet->getML(i)->getInputnum();
			if(i == 0)
				tempconvinputData.img_data = input;
			else
				tempconvinputData.img_data = mynet->getML(i-1)->getConvOutdata();
			
			tempconvoutputData.height = mynet->getML(i)->getOutputImageSize();
			tempconvoutputData.img_num = mynet->getML(i)->getOutputnum();
			tempconvoutputData.img_data = mynet->getML(i)->getConvOutdata();
			
			tempconvweightData.inputNum = mynet->getML(i)->getInputnum();
			tempconvweightData.outputNum = mynet->getML(i)->getOutputnum();
			tempconvweightData.kernelSize = mynet->getML(i)->getKernelSize();
			tempconvweightData.bias.dataSize = mynet->getML(i)->getOutputnum();
			tempconvweightData.bias.data = mynet->getML(i)->getConvBias();
			tempconvweightData.weight = mynet->getML(i)->getConvWeight();
			Forward_conv(1, tempconvinputData, tempconvweightData, &tempconvoutputData);
		   //print_mat("after output image",tempconvoutputData.img_data[0],tempconvoutputData.height,tempconvoutputData.height);
		}
		else if(mynet->getML(i)->getLayerType() == POOL){
			tempconvinputData.height = mynet->getML(i)->getInputImageSize();
			tempconvinputData.img_num = mynet->getML(i)->getInputnum();
			tempconvinputData.img_data = mynet->getML(i-1)->getConvOutdata();

			tempconvoutputData.height = mynet->getML(i)->getOutputImageSize();
			tempconvoutputData.img_num = mynet->getML(i)->getOutputnum();
			tempconvoutputData.img_data = mynet->getML(i)->getConvOutdata();
			Forward_pool(tempconvinputData, 2, 1, &tempconvoutputData);
			//print_mat("after",tempconvoutputData.img_data[0],tempconvoutputData.height,tempconvoutputData.height);
		}
		else if(mynet->getML(i)->getLayerType() == FULLCON){
			tempinputData.dataSize = mynet->getML(i)->getInputnum();
			tempconvinputData.img_data = mynet->getML(i-1)->getConvOutdata();
			tempinputData.data = new float[tempinputData.dataSize];
			int outsize = mynet->getML(i-1)->getOutputImageSize();
			for (int k = 0; k < (mynet->getML(i-1)->getOutputnum()); k++)
				for (int r = 0; r < outsize; r++)
					for (int c = 0; c < outsize; c++)
						tempinputData.data[k * outsize * outsize + r * outsize + c] = tempconvinputData.img_data[k][r][c];
		
			tempweightData.inputNum = mynet->getML(i)->getInputnum();
			tempweightData.outputNum = mynet->getML(i)->getOutputnum(); 		
			tempweightData.weight = mynet->getML(i)->getWeight();

			tempoutputData.dataSize = mynet->getML(i)->getOutputnum();
			tempoutputData.data = mynet->getML(i)->getOutdata();
			Forward(tempinputData, tempweightData, &tempoutputData);
			//print_vector("after forward",tempoutputData.data,tempoutputData.dataSize);
			//delete(tempinputData.data);
		}
	}
	int max = 0;
	for(int k = 1;k<tempoutputData.dataSize;k++){
		if(tempoutputData.data[k]>tempoutputData.data[max])
			max = k;
	}
	printf("expect is %d,we get %d\n",result,max);
	if(max == result)
		return 0;
	else
		return 1;
	
}

void test( MyNet *mynet, ParamBuf *param)
{
	FILE *fp,*fpLabel;
	u8 buffer[16]={0};
	u8 buffer_label[8]={0};
	u8 image_buffer[28][28]={0};
#ifdef USE_MALLOC_TEMP
	float ***trains_image;
#else
	float ***trains_image;
    float trains_image_arr[1][28][28];
	float **trains_image_pp[1];
	float *trains_image_p[28];
	for(int i = 0;i<28;i++){
		trains_image_p[i] = trains_image_arr[0][i];
	}
	trains_image_pp[0] = trains_image_p;
	trains_image = trains_image_pp;
#endif

	u8 label = 0;
	u8 i = 0;
	u32 count = 0;
	u32 error = 0;
	if ((fp=fopen("./images/t10k-images.idx3-ubyte","rb"))==NULL){
		printf("open failed\n");
		return;
	}
	if ((fpLabel=fopen("./images/t10k-labels.idx1-ubyte","rb"))==NULL){
		printf("打开文件错误\n");
		return;
	}
	fread(buffer_label,1,sizeof(buffer_label),fpLabel);
	for(i=0;i<=8;i++)
	{
		printf("%x ",buffer_label[i]);
	}
	fread(buffer,1,sizeof(buffer),fp);
	for(i=0;i<=15;i++)
	{
		printf("%x ",buffer[i]);
	}
	printf("\n");
#ifdef USE_MALLOC_TEMP    
	trains_image = new float **[1];
	trains_image[0] = new float *[28];
	for(i = 0;i<28;i++)
		trains_image[0][i] = new float [28];
#endif
	while(fread(image_buffer,1,sizeof(image_buffer),fp)>0 && fread(&label,1,1,fpLabel)>0)
	{
		{
			u8 i,j;
			printf("testing %d images:",++count);
			
			for(i = 0;i<=27;i++){
				for(j = 0;j<=27;j++){
					trains_image[0][i][j] = (float)image_buffer[i][j]/255;	
					//printf("%3d ",image_buffer[i][j]);
				}
				//printf("\n");
			}
		}
		for(int l = 0;l<mynet->getLayerNum();l++)
			mynet->getML(l)->clearData();
		if(mytest(mynet, (float ***)trains_image, label))
			error++;
		
	}
	printf("\ntotal test %d,error %d, accuracy:%f\n",count,error,(float)((float)(count-error)/(float)count));
	


}


void train( MyNet *mynet, ParamBuf *param, int isFPGA)
{
    int fd,fd11;

    if (isFPGA == 1)
    {
        fd11 = open("/dev/neural_network_accel_core", O_RDWR);
        if (fd11 == -1)
        {
            printf("Can't open /dev/neuralacc. in FL\n");
        }

        fd = open("/dev/dma", O_RDWR);
        if (fd == -1)
        {
            printf("Can't open /dev/dma. in FL\n");
        }
    }

    FILE *fp,*fpLabel;
    u8 buffer[16]={0};
	u8 buffer_label[8]={0};
    u8 image_buffer[28][28]={0};
#ifdef USE_MALLOC_TEMP
	float ***trains_image;
#else
	float ***trains_image;
    float trains_image_arr[1][28][28];
	float **trains_image_pp[1];
	float *trains_image_p[28];
	for(int i = 0;i<28;i++){
		trains_image_p[i] = trains_image_arr[0][i];
	}
	trains_image_pp[0] = trains_image_p;
	trains_image = trains_image_pp;
#endif
	float result_output[10] = {0.0};
	u8 label = 0;
    u8 i = 0;
    u32 count = 0;
    if ((fp=fopen(param->getImagePath().c_str(),"rb"))==NULL){
        printf("open failed\n");
        return;
    }
    if ((fpLabel=fopen("./images/train-labels.idx1-ubyte","rb"))==NULL){
        printf("打开文件错误\n");
        return;
    }
    fread(buffer_label,1,sizeof(buffer_label),fpLabel);
    for(i=0;i<=8;i++)
    {
        printf("%x ",buffer_label[i]);
    }
    fread(buffer,1,sizeof(buffer),fp);
    for(i=0;i<=15;i++)
    {
        printf("%x ",buffer[i]);
    }
    printf("\n");
#ifdef USE_MALLOC_TEMP    
    trains_image = new float **[1];
    trains_image[0] = new float *[28];
    for(i = 0;i<28;i++)
        trains_image[0][i] = new float [28];
#endif
    while(fread(image_buffer,1,sizeof(image_buffer),fp)>0 && fread(&label,1,1,fpLabel)>0)
    {
        {
            u8 i,j;
			printf("handling %d images:\n",++count);
            for(i = 0;i<=27;i++){
                for(j = 0;j<=27;j++){
					trains_image[0][i][j] = (float)image_buffer[i][j]/255;	
                    //printf("%3d ",image_buffer[i][j]);
                }
                //printf("\n");
            }
            printf("The actual result is %d\n",label);
			for(i=0;i<10;i++)
				result_output[i] = (float)0.0;
			result_output[label] = (float)1.0;
        }
        for(int l = 0;l<mynet->getLayerNum();l++)
            mynet->getML(l)->clearData();
		Step(mynet, (float ***)trains_image, result_output, isFPGA, fd, fd11);
        
        //if(count==3)
          // break;
         //return;
    }
    
    //test( mynet, &input[0] );
    if (isFPGA == 1)
    {
        close(fd);
        close(fd11);
    }
}

int main(int argc, char* argv[])
{
    google::InitGoogleLogging(argv[0]);
    google::LogToStderr();   

    SolverParameter SGD;
    int i;
   
    google::SetUsageMessage("command line brew\n"
    "usage: caffe <command> <args>\n\n"
    "commands:\n"
    "  train           train or finetune a model\n"
    "  test            score a model\n");

    if (argc != 2)
    {
        google::ShowUsageWithFlagsRestrict(argv[0], "caffe");
        return -1;
    }

    ParamBuf param;

    if(!ReadProtoFromTextFile("lenet_solver.prototxt", &SGD))
    {
        cout<<"error opening file"<<endl; 
        return -1;
    }
    LOG(INFO) << "This is the first test !";

    LOG(INFO)<<"hello, this is a test file of reading the params and store it.";
    param.setImagePath((string)SGD.net());
    param.setLearnRate(SGD.base_lr());
    param.setMax_Iterm(SGD.max_iter());
    //cout<<SGD.test_iter()<<endl;
    LOG(INFO)<<param.getImagePath();
    LOG(INFO)<<param.getLearnRate();
    LOG(INFO)<<param.getMax_iterm();
    LOG(INFO)<<"End reading lenet_solver";

    NetParameter VGG16;
    if(!ReadProtoFromTextFile("lenet.prototxt", &VGG16))
    {
        cout<<"error opening file"<<endl; 
        return -1;
    }

    MyNet mynet = MyNet(VGG16.layer_size()-1, VGG16.name());
    LOG(INFO)<<mynet.getLayerNum();
    LOG(INFO)<<mynet.getNetname();

    string Layernamebuffer;
    int inputnum=1;
    int outputnum;
	int imageW = 0;
	int outimageW = 0;
	int kernelsize = 0;
	LayerType type;

    for (i=0;i<VGG16.layer_size();i++)
    {
        Layernamebuffer = VGG16.layer(i).name();
        LOG(INFO)<<Layernamebuffer;
        if (i == 0)
        {
        	if(string(VGG16.layer(i).type())==string("Input"))
        	{
				//imageW = VGG16.layer(i).input_param().shape().dim_size();
				//imageH = VGG16.layer(i).input_param().shape().dim_size();
                outimageW = 28;
                LOG(INFO)<<imageW;
			}
			else
			{
				cout<<"input layer error"<<endl; 
				return 0;
			}
            continue;
        }
		if(string(VGG16.layer(i).type())==string("Convolution"))
		{
			if(i != 1)
				inputnum = VGG16.layer(i-2).convolution_param().num_output();
			outputnum = VGG16.layer(i).convolution_param().num_output();
			//kernersize = VGG16.layer(i).convolution_param().kernel_size();
			kernelsize = 5;
			imageW = outimageW;
            outimageW = imageW - kernelsize + 1;
			type = CONVOL;
		}
		else if(string(VGG16.layer(i).type())==string("Pooling"))
		{
			inputnum = VGG16.layer(i-1).convolution_param().num_output();
			outputnum = inputnum;			
			imageW = outimageW;
			outimageW = imageW / 2;
			type = POOL;
		}
		else if(string(VGG16.layer(i).type())==string("InnerProduct"))
		{
			inputnum = outimageW * outimageW * VGG16.layer(i-2).convolution_param().num_output();
			outputnum = VGG16.layer(i).inner_product_param().num_output();
			type = FULLCON;
			imageW = 1;
			outimageW = 1;
		}
        
        //LOG(INFO)<<outputnum;
        mynet.getML(i-1)->setInputnum(inputnum);
        mynet.getML(i-1)->setOutputnum(outputnum);
        mynet.getML(i-1)->setLayerName(Layernamebuffer);
		mynet.getML(i-1)->setLayerType(type);
		mynet.getML(i-1)->setInputImageSize(imageW);
		mynet.getML(i-1)->setOutputImageSize(outimageW);
		mynet.getML(i-1)->setKernelSize(kernelsize);
    }
	for (i=0;i<VGG16.layer_size()-1;i++)
	{
        LOG(INFO)<<mynet.getML(i)->getLayerName();
		LOG(INFO)<<mynet.getML(i)->getInputnum();
        LOG(INFO)<<mynet.getML(i)->getOutputnum();
		LOG(INFO)<<mynet.getML(i)->getInputImageSize();
        LOG(INFO)<<mynet.getML(i)->getOutputImageSize();
	}
    LOG(INFO)<<"End reading lenet";
	
    GlobalInit( &mynet );

	malloc_temp_buffer();

    if (strcmp(argv[1], "FPGA_train") == 0)
    {
        LOG(INFO) << "Begin training!";
        train(&mynet, &param, 1);
        LOG(INFO) << "End trainging!";
		LOG(INFO) << "Begin testing!";
        test(&mynet, &param);
        LOG(INFO) << "End testing!";
    }
    else if (strcmp(argv[1], "train") == 0)
    {
        LOG(INFO) << "Begin training!";
        train(&mynet, &param, 0);
        LOG(INFO) << "End trainging!";
		LOG(INFO) << "Begin testing!";
        test(&mynet, &param);
        LOG(INFO) << "End testing!";
    }


    //for (int j = 0 ; j < mynet.getML(0)->getOutputnum(); j++)
    //cout<<mynet.getML(0)->getOutdata()[j]<<endl;
    //cout<<mynet.getML(1)->getOutdata()<<endl;

    return 0;
}
