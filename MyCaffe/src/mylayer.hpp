#include <string>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>
using namespace std;
enum LayerType{
	CONVOL = 0,
	POOL = 1,
	FULLCON = 2
};
class MyLayer{
private:

	string layername;

	LayerType layertype;
		
    int inputnum;

    int outputnum;

    int gradientnum;

	int inputimagesize;

	int outputimagesize;

	int kernelsize;

	float *gradient;

	float *outdata;

	float **weigth;

	float ***convgradient;

	float ***convoutdata;

	float ****convweight;

	float *convbias;

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
	void setLayerType(LayerType type){
		layertype = type;
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

	void setInputImageSize(int size){
		inputimagesize = size;
	}
	
	void setOutputImageSize(int size){
		outputimagesize = size;
	} 
	void GradientInit(){
		if(layertype == FULLCON){
        	gradient = new float[outputnum];
            for(int i = 0;i<outputnum;i++)
                gradient[i] = (float)0.0;
        }
		else{
			convgradient = new float **[outputnum];
            float **x = new float *[outputnum*outputimagesize];
            float *y = new float [outputnum*outputimagesize*outputimagesize];

			for(int i = 0;i<outputnum;i++){
				convgradient[i] = &(x[i*outputimagesize]);
				for(int r=0;r<outputimagesize;r++){
					x[i*outputimagesize+r] = &(y[i*outputimagesize*outputimagesize+r*outputimagesize]);
                    for(int c = 0;c<outputimagesize;c++)
                        convgradient[i][r][c] = (float)0.0;
			    }
		    }
        }
    }

	void setGradient(float *g , int size){
		for(int i=0 ; i<size ; i++){
			gradient[i] = g[i];
		}
	}
	
	void setConvGradient(float ***g , int num,int row,int col){
		for(int i=0 ; i<num ; i++){
			for(int r=0; r<row;r++)
				for(int c=0; c<col;c++)
					convgradient[i][r][c] = g[i][r][c];
		}
	}

    void OutdataInit(){
    	if(layertype == FULLCON){
			outdata = new float[this->outputnum];
            for(int i = 0;i<outputnum;i++)
                outdata[i] = (float)0.0;
        }
		else{
			convoutdata = new float **[outputnum];
            float **x = new float *[outputnum*outputimagesize];
            float *y = new float [outputnum*outputimagesize*outputimagesize];
			for(int i = 0;i<outputnum;i++){
				convoutdata[i] = &(x[i*outputimagesize]);
				for(int r=0;r<outputimagesize;r++){
					x[i*outputimagesize+r] = &(y[i*outputimagesize*outputimagesize+r*outputimagesize]);
                    for(int c = 0;c<outputimagesize;c++)
                        convoutdata[i][r][c] = (float)0.0;
			    }
		    }
            }
    }

	void setOutdata(float *o , int size){
		for(int i=0 ; i<size ; i++){
			outdata[i] = o[i];
		}
	}

	void setConvOutdata(float ***o , int num,int row,int col){
		for(int i=0 ; i<num ; i++){
			for(int r=0; r<row;r++)
				for(int c=0; c<col;c++)
					convoutdata[i][r][c] = o[i][r][c];
		}
	}

    void WeightInit(){
		if(layertype == POOL)
			return;
		else if(layertype == FULLCON)
		{
			weigth = new float *[outputnum];
		    for (int i = 0; i < outputnum; i++)
		    {
			    weigth[i] = new float[inputnum+1];
				for(int j = 0;j<inputnum+1;j++){
					float randnum = (((float)rand() / (float)RAND_MAX) - 0.5) * 2;
					weigth[i][j] = randnum * sqrt((float)6.0 / (float)(outputnum + inputnum));
                    //weigth[i][j] = 1.0;
				}
	        }
		}
		else{
			convweight = new float ***[outputnum];
			for(int i = 0;i<outputnum;i++){
				convweight[i] = new float **[inputnum];
				for(int j = 0;j<inputnum;j++){
					convweight[i][j] = new float *[kernelsize];
					for(int r=0;r<kernelsize;r++){
						convweight[i][j][r] = new float [kernelsize]; //not initial
						for(int c=0; c<kernelsize; c++){
							float randnum = (((float)rand() / (float)RAND_MAX) - 0.5) * 2;
							convweight[i][j][r][c] = randnum * sqrt((float)6.0 / (float)(kernelsize * kernelsize * (outputnum + inputnum)));
						    //convweight[i][j][r][c] = 1.0;
                        }
                    }		
			    }
            }
			convbias = new float[outputnum];
            for(int i = 0;i<outputnum;i++)
                convbias[i] = (float)0.1;
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

	void setConvWeight(float ****w , const int onum,const int inum,const int row , const int col){
		for (int i = 0; i < onum; i++)
		{
			for(int j=0 ; j<inum ; j++){
				for (int r = 0; r < row; r++)
				{
					for(int c=0 ; c<col ; c++){
						convweight[i][j][r][c] = w[i][j][r][c];
				    }
			    }
            }
        }
    }

	void clearData(){
		if(layertype == FULLCON)
		{
			for(int i = 0;i<outputnum;i++){
                outdata[i] = (float)0.0;
                gradient[i] = (float)0.0;
			}
		}
		else{
			for(int i = 0;i<outputnum;i++)
				for(int r=0;r<outputimagesize;r++)
                    for(int c = 0;c<outputimagesize;c++){
                        convoutdata[i][r][c] = (float)0.0;
                        convgradient[i][r][c] = (float)0.0;
			    }
		    
		}
	}

	void setKernelSize(int size){
		kernelsize = size;
	}

	int getKernelSize(){
		return kernelsize;
	}
	float *getGradient(){
		return this->gradient;
	}

	float ***getConvGradient(){
		return this->convgradient;
	}

	float *getOutdata(){
		return this->outdata;
	}

	float ***getConvOutdata(){
		return this->convoutdata;
	}

	float **getWeight(){
		return this->weigth;
	}

	float ****getConvWeight(){
		return this->convweight;
	}

	float *getConvBias(){
		return convbias;
	}
    string getLayerName(){
        return layername;
    }

	LayerType getLayerType(){
		return layertype;
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

	int getInputImageSize(){
		return inputimagesize;
	}

	int getOutputImageSize(){
		return outputimagesize;
	}
};
