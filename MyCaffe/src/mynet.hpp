
#include <string>
#include "mylayer.hpp"
using namespace std;

class MyNet{
private:
	int layernum;
	string netname;
	MyLayer *ml;

public:
	MyNet():layernum(0){
		netname = "";
		ml = NULL;
	}


	MyNet(int ln , string name):layernum(ln),netname(name){
		ml = new MyLayer[ln];
	}

	int getLayerNum(){
		return this->layernum;
	}

	MyLayer* getML(int index){
		return &((this->ml)[index]);
	}

    string getNetname(){
        return this->netname;
    }
};
