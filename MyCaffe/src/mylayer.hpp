#include <string>
using namespace std;

class MyLayer{
private:

	string layername;

    int inputnum;

    int outputnum;

    int gradientnum;

	float *gradient;

	float *outdata;

	float **weigth;

public:

	MyLayer(){
		layername = "";
		gradient = NULL;
		outdata = NULL;
		weigth = NULL;
	}

	MyLayer(string str, int inputnum, int outputnum):
    inputnum(inputnum),outputnum(outputnum){
		layername = str;
		gradient = NULL;
		outdata = NULL;
		weigth = NULL;
	}

	void setLayerName(string &name){
		layername = name;
	}

    void setInputnum(int input){
        inputnum = input;
    }

    void setOutputnum(int output){
        outputnum = output;
    }

    void setGradientnum(int gradient){
        gradientnum = gradient;
    }

    void GradientInit(){
        gradient = new float[this->outputnum];
    }

	void setGradient(float *g , int size){
		for(int i=0 ; i<size ; i++){
			gradient[i] = g[i];
		}
	}

    void OutdataInit(){
		outdata = new float[this->outputnum];
    }

	void setOutdata(float *o , int size){
		for(int i=0 ; i<size ; i++){
			outdata[i] = o[i];
		}
	}

    void WeightInit(){
		weigth = new float *[outputnum];
        {
		    for (int i = 0; i < outputnum; i++)
		    {
			    weigth[i] = new float[inputnum+1];
	        }
        }
    }

	void setWeight(float **w , const int row , const int col){
		for (int i = 0; i < row; i++)
		{
			for(int j=0 ; j<col ; j++){
				weigth[i][j] = w[i][j];
		    }
	    }
    }

	float *getGradient(){
		return this->gradient;
	}

	float *getOutdata(){
		return this->outdata;
	}

	float **getWeight(){
		return this->weigth;
	}

    string getLayerName(){
        return layername;
    }

    int getInputnum(){
        return inputnum;
    }

    int getOutputnum(){
        return outputnum;
    }

    int getGradientnum(){
        return gradientnum;
    }
};
