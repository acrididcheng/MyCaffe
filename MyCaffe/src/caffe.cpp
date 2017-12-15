#include <fcntl.h>
#include <unistd.h>

#include <iostream>
#include <string>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <glog/logging.h>
#include <gflags/gflags.h>

#include "caffe.pb.h"
#include "parambuf.hpp"
#include "mynet.hpp"
#include "compute.h"
using namespace caffe;
using namespace std;
#define INPUT_NUM 5
#define OUTPUT_NUM 5
#define DATA_NUM 1

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

void Step( MyNet *mynet, float *input, float *output)
{
    int i;

    //Forward procedure
    ioData tempinputData,tempoutputData,tempData;
    weightData tempweightData;
    gradientData tempgradientData,tempgradientDataOut;

    //first time to operate the tempinputData
    tempinputData.dataSize = INPUT_NUM;
    tempinputData.data = input;
    tempweightData.inputNum = tempinputData.dataSize;

    for (i = 0; i < mynet->getLayerNum(); i++)
    {
        if (i != 0 )
        {
            tempweightData.inputNum = mynet->getML(i-1)->getOutputnum();
            tempinputData.dataSize = mynet->getML(i-1)->getOutputnum();
            tempinputData.data = mynet->getML(i-1)->getOutdata();
        }
        tempweightData.outputNum = mynet->getML(i)->getOutputnum();
        //allocate space for the weightdata
        
        tempweightData.weight = mynet->getML(i)->getWeight();

        tempoutputData.dataSize = mynet->getML(i)->getOutputnum();
        tempoutputData.data = mynet->getML(i)->getOutdata();

        Forward(tempinputData, tempweightData, &tempoutputData);
        
        //mynet->getML(i)->setOutdata(tempoutputData.data,mynet->getML(i)->getOutputnum());

    }

    tempinputData.dataSize = OUTPUT_NUM;
    tempinputData.data = output;

    tempData.dataSize = mynet->getML(mynet->getLayerNum()-1)->getOutputnum();
    tempData.data = mynet->getML(mynet->getLayerNum()-1)->getOutdata();

    tempgradientData.gradientSize = mynet->getML(mynet->getLayerNum()-1)->getOutputnum();
    tempgradientData.gradient = mynet->getML(mynet->getLayerNum()-1)->getGradient();

    OutputGradient( tempinputData, tempData, &tempgradientData );
    //mynet->getML(mynet->getLayerNum()-1)->setGradientnum(tempgradientData.gradientSize);
    //mynet->getML(mynet->getLayerNum()-1)->setGradient(tempgradientData.gradient, tempgradientData.gradientSize);

    for (i = mynet->getLayerNum() - 1; i > 0; i--)
    {
        tempgradientData.gradientSize = mynet->getML(i)->getOutputnum();
        tempgradientData.gradient = mynet->getML(i)->getGradient();

        tempweightData.inputNum = mynet->getML(i)->getInputnum();
        tempweightData.outputNum = mynet->getML(i)->getOutputnum();
        tempweightData.weight = mynet->getML(i)->getWeight();

        tempoutputData.dataSize = mynet->getML(i)->getOutputnum();
        tempoutputData.data = mynet->getML(i)->getOutdata();
    
        tempgradientDataOut.gradientSize = mynet->getML(i-1)->getOutputnum();
        tempgradientDataOut.gradient = new float[tempgradientDataOut.gradientSize];

        Backward( tempgradientData, tempweightData, tempoutputData, &tempgradientDataOut);

        //mynet->getML(i-1)->setGradientnum(tempgradientDataOut.gradientSize);
        mynet->getML(i-1)->setGradient(tempgradientDataOut.gradient,tempgradientDataOut.gradientSize);     
    }

    for (i = 0; i < mynet->getLayerNum(); i++)
    {
        tempgradientData.gradientSize = mynet->getML(i)->getOutputnum();
        tempgradientData.gradient = mynet->getML(i)->getGradient();
        
        tempData.dataSize = mynet->getML(i)->getInputnum();
        if (i != 0)
            tempData.data = mynet->getML(i-1)->getOutdata();
        else
            tempData.data = input;

        tempweightData.inputNum = mynet->getML(i)->getInputnum();
        tempweightData.outputNum = mynet->getML(i)->getOutputnum();
        tempweightData.weight = mynet->getML(i)->getWeight();
        
        ApplyUpdate(tempgradientData, tempData, &tempweightData);
    }

}

void test( MyNet *mynet, float *input)
{
    int i; 
    ioData tempinputData,tempoutputData;
    weightData tempweightData;

    //first time to operate the tempinputData
    tempinputData.dataSize = INPUT_NUM;
    tempinputData.data = input;
    tempweightData.inputNum = tempinputData.dataSize;

    for (i = 0; i < mynet->getLayerNum(); i++)
    {
        if (i != 0 )
        {
            tempweightData.inputNum = mynet->getML(i-1)->getOutputnum();
            tempinputData.dataSize = mynet->getML(i-1)->getOutputnum();
            tempinputData.data = mynet->getML(i-1)->getOutdata();
        }
        tempweightData.outputNum = mynet->getML(i)->getOutputnum();
        //allocate space for the weightdata
        
        tempweightData.weight = mynet->getML(i)->getWeight();

        tempoutputData.dataSize = mynet->getML(i)->getOutputnum();
        tempoutputData.data = mynet->getML(i)->getOutdata();

        Forward(tempinputData, tempweightData, &tempoutputData);
        
        //mynet->getML(i)->setOutdata(tempoutputData.data,mynet->getML(i)->getOutputnum());
    }


    cout << "INPUT:";
    for (i = 0; i < INPUT_NUM; i++)
    {
        cout << input[i] << ",";
    }
    cout << endl;

    cout << "OUTPUT:";
    for (i = 0; i < OUTPUT_NUM; i++)
    {
        cout << mynet->getML(mynet->getLayerNum()-1)->getOutdata()[i] << ",";
    }
    cout << endl;

}

void train( MyNet *mynet, float *input, float *output, ParamBuf *param)
{
    int i;
    srand((unsigned)time(0));
    for (i = 0; i < (int)param->getMax_iterm(); i++)
    {
        Step(mynet, &input[rand()%DATA_NUM], &output[rand()%DATA_NUM]);
    }
    
    test( mynet, &input[0] );
}

int main(int argc, char* argv[])
{
    google::InitGoogleLogging(argv[0]);
    google::LogToStderr();   

    SolverParameter SGD;
    int i;

    gflags::SetUsageMessage("command line brew\n"
    "usage: caffe <command> <args>\n\n"
    "commands:\n"
    "  train           train or finetune a model\n"
    "  test            score a model\n");

    if (argc != 2)
    {
        gflags::ShowUsageWithFlagsRestrict(argv[0], "caffe");
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

    MyNet mynet = MyNet(VGG16.layer_size(), VGG16.name());
    LOG(INFO)<<mynet.getLayerNum();
    LOG(INFO)<<mynet.getNetname();

    string Layernamebuffer;
    int inputnum=5;
    int outputnum;

    for (i=0;i<VGG16.layer_size();i++)
    {
        Layernamebuffer = VGG16.layer(i).name();
        if (i != 0)
            inputnum = VGG16.layer(i-1).inner_product_param().num_output();
        outputnum = VGG16.layer(i).inner_product_param().num_output();
        LOG(INFO)<<outputnum;
        mynet.getML(i)->setInputnum(inputnum);
        mynet.getML(i)->setOutputnum(outputnum);
        mynet.getML(i)->setLayerName(Layernamebuffer);
    }


    LOG(INFO)<<"End reading lenet";

    GlobalInit( &mynet );

    if (strcmp(argv[1], "train") == 0)
    {
        LOG(INFO) << "Begin training!";
        train(&mynet, &inputmatrix[0][0], &outputmatrix[0][0], &param);
        LOG(INFO) << "End trainging!";
    }
    //for (int j = 0 ; j < mynet.getML(0)->getOutputnum(); j++)
    //cout<<mynet.getML(0)->getOutdata()[j]<<endl;
    //cout<<mynet.getML(1)->getOutdata()<<endl;

    return 0;
}
