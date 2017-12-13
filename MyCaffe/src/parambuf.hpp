#include <string>
using namespace std;

class ParamBuf{
private:
	string imagePath;

	float learnRate;

	unsigned int max_iterm;

public:
	ParamBuf(){
		learnRate = 0;
		max_iterm = 0;
	}

	ParamBuf(string path , float lr , unsigned int max){
		imagePath = path;
		learnRate = lr;
		max_iterm = max;
	}

	void setImagePath(string path){
		this->imagePath = path;
	}

	void setLearnRate(float lr){
		this->learnRate = lr;
	}

	void setMax_Iterm(unsigned int max){
		this->max_iterm = max;
	}

	string getImagePath(){
		return this->imagePath;
	}

	float getLearnRate(){
		return this->learnRate;
	}

	unsigned int getMax_iterm(){
		return this->max_iterm;
	}

	~ParamBuf(){}
};
