#include <fcntl.h>
#include <unistd.h>

#include <iostream>
#include <string>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include "caffe.pb.h"
using namespace caffe;
using namespace std;

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


int main()
{
    SolverParameter SGD;
    int i;

    if(!ReadProtoFromTextFile("lenet_solver.prototxt", &SGD))
    {
        cout<<"error opening file"<<endl; 
        return -1;
    }

    cout<<"hello,world"<<endl;
    cout<<SGD.net()<<endl;
    //cout<<SGD.test_iter()<<endl;
    cout<<SGD.test_interval()<<endl;
    cout<<SGD.base_lr()<<endl;
    cout<<SGD.momentum()<<endl;
    cout<<"End reading lenet_solver"<<endl;

    NetParameter VGG16;
    if(!ReadProtoFromTextFile("lenet.prototxt", &VGG16))
    {
        cout<<"error opening file"<<endl; 
        return -1;
    }
    cout<<VGG16.name()<<endl;
    cout<<VGG16.layer_size()<<endl;
    for ( i =0 ; i < VGG16.layer_size() ; i++ )
    {
        cout<<VGG16.layer(i).name()<<endl;
    }
    cout<<"End reading lenet"<<endl;
    return 0;
}
