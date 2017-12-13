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

	void setGradient(float *g , int size){
		gradient = new float[size];
		for(int i=0 ; i<size ; i++){
			gradient[i] = g[i];
		}
	}

	void setOutdata(float *o , int size){
		outdata = new float[size];
		for(int i=0 ; i<size ; i++){
			outdata[i] = o[i];
		}
	}

	void setWeight(float **w , const int row , const int col){
		weigth = new float *[row];

        {
		    for (int i = 0; i < row; i++)
		    {
			    weigth[i] = new float[col];
			    for(int j=0 ; j<col ; j++){
                    if ( w!= NULL )
                    {
				        weigth[i][j] = w[i][j];

			        }
		        }
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
