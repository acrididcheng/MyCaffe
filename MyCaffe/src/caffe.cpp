#include <fcntl.h>
#include <unistd.h>

#include <iostream>
#include <string>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include "caffe.pb.h"
#include "parambuf.hpp"
#include "mynet.hpp"
#include "compute.h"
using namespace caffe;
using namespace std;
#define INPUT_NUM 5
#define OUTPUT_NUM 5

float inputmatrix[][5]=
{
    {0.1,0.5,0.8,0.9,1.0},
    {0.2,0.4,0.1,0.5,0.4},
    {0.2,0.1,0.5,0.3,0.5}
};

float outputmatrix[][5]=
{
    {0.1,0.5,0.8,0.9,1.0},
    {0.2,0.4,0.1,0.5,0.4},
    {0.2,0.1,0.5,0.3,0.5}
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
        mynet->getML(i)->setWeight(NULL,mynet->getML(i)->getOutputnum(),mynet->getML(i)->getInputnum()+1);
        
        tempweightData.weight = mynet->getML(i)->getWeight();

        tempoutputData.dataSize = mynet->getML(i)->getOutputnum();
        tempoutputData.data = new float[tempoutputData.dataSize];

        Forward(tempinputData, tempweightData, &tempoutputData);
        
        mynet->getML(i)->setOutdata(tempoutputData.data,mynet->getML(i)->getOutputnum());
    }

    tempinputData.dataSize = OUTPUT_NUM;
    tempinputData.data = output;

    tempData.dataSize = mynet->getML(mynet->getLayerNum()-1)->getOutputnum();
    tempData.data = mynet->getML(mynet->getLayerNum()-1)->getOutdata();

    tempgradientData.gradientSize = mynet->getML(mynet->getLayerNum()-1)->getOutputnum();
    tempgradientData.gradient = new float[tempgradientData.gradientSize];

    OutputGradient( tempinputData, tempData, &tempgradientData );
    mynet->getML(mynet->getLayerNum()-1)->setGradientnum(tempgradientData.gradientSize);
    mynet->getML(mynet->getLayerNum()-1)->setGradient(tempgradientData.gradient, tempgradientData.gradientSize);

    for (i = mynet->getLayerNum() - 1; i > 0; i--)
    {
        tempgradientData.gradientSize = mynet->getML(i)->getGradientnum();
        tempgradientData.gradient = mynet->getML(i)->getGradient();

        tempweightData.inputNum = mynet->getML(i)->getInputnum();
        tempweightData.outputNum = mynet->getML(i)->getOutputnum();
        tempweightData.weight = mynet->getML(i)->getWeight();

        tempoutputData.dataSize = mynet->getML(i)->getOutputnum();
        tempoutputData.data = mynet->getML(i)->getOutdata();
    
        tempgradientDataOut.gradientSize = mynet->getML(i-1)->getOutputnum();
        tempgradientDataOut.gradient = new float[tempgradientDataOut.gradientSize];

        Backward( tempgradientData, tempweightData, tempoutputData, &tempgradientDataOut);

        mynet->getML(i-1)->setGradientnum(tempgradientDataOut.gradientSize);
        mynet->getML(i-1)->setGradient(tempgradientDataOut.gradient,tempgradientDataOut.gradientSize);     
    }

    for (i = 0; i < mynet->getLayerNum(); i++)
    {
        tempgradientData.gradientSize = mynet->getML(i)->getGradientnum();
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



int main()
{
    SolverParameter SGD;
    int i;
    int num_input=5;

    ParamBuf param;

    if(!ReadProtoFromTextFile("lenet_solver.prototxt", &SGD))
    {
        cout<<"error opening file"<<endl; 
        return -1;
    }

    cout<<"hello, this is a test file of reading the params and store it."<<endl;
    param.setImagePath((string)SGD.net());
    param.setLearnRate(SGD.base_lr());
    param.setMax_Iterm(SGD.max_iter());
    //cout<<SGD.test_iter()<<endl;
    cout<<param.getImagePath()<<endl;
    cout<<param.getLearnRate()<<endl;
    cout<<param.getMax_iterm()<<endl;
    cout<<"End reading lenet_solver"<<endl;

    NetParameter VGG16;
    if(!ReadProtoFromTextFile("lenet.prototxt", &VGG16))
    {
        cout<<"error opening file"<<endl; 
        return -1;
    }

    MyNet mynet = MyNet(VGG16.layer_size(), VGG16.name());
    cout<<mynet.getLayerNum()<<endl;
    cout<<mynet.getNetname()<<endl;

    string Layernamebuffer;
    int inputnum=5;
    int outputnum;

    for (i=0;i<VGG16.layer_size();i++)
    {
        Layernamebuffer = VGG16.layer(i).name();
        if (i != 0)
            inputnum = VGG16.layer(i-1).inner_product_param().num_output();
        outputnum = VGG16.layer(i).inner_product_param().num_output();
        cout<<outputnum<<endl;
        mynet.getML(i)->setInputnum(inputnum);
        mynet.getML(i)->setOutputnum(outputnum);
        mynet.getML(i)->setLayerName(Layernamebuffer);
    }


    cout<<"End reading lenet"<<endl;

    mynet.getML(0)->setOutdata(outputmatrix[1],sizeof(outputmatrix[1])/4);


    Step(&mynet, inputmatrix[0], outputmatrix[0]);
    //for (int j = 0 ; j < mynet.getML(0)->getOutputnum(); j++)
    //cout<<mynet.getML(0)->getOutdata()[j]<<endl;
    //cout<<mynet.getML(1)->getOutdata()<<endl;

    return 0;
}
