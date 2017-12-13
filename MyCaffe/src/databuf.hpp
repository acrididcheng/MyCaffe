class Picture{
private:
	int channel;
	int heigth;
	int width;
public:
	Picture():channel(0),heigth(0),width(0){}

	Picture(int c , int h , int w){
		channel = c;
		heigth = h;
		width = w;
	}

	~Picture(){}
};


class DataBuf{
private:
	int sampleCount;	//总的样本的数
	int inum;			//用于输入的节点数
	int onum;			//用于输出的节点数
	float lrate;		//学习率
	double *gradient;	//梯度
	double *outdata;	//输出的数据
	Picture *pict;
	double **weigth;	//权重
public:
	/*
	 * 构造函数，并且进行初始化
	 */
	DataBuf(int s , float lr , double **w , double *g , double *o , Picture *p)
		:sampleCount(s),lrate(lr){
		inum = sizeof(g);
		onum = sizeof(o);

			//初始化gradient
		gradient = new double[inum];
		for(int i=0 ; i<inum ; i++){
			gradient[i] = g[i];
		}
		//初始化outdata
		outdata = new double[onum];
		for(int i=0 ; i<onum ; i++){
			outdata[i ] = o[i];
		}
		//初始化weight
		weigth = new double *[inum];
		for(int i=0 ; i<inum ; i++){
			weigth[i] = new double[onum];
		}
		for(int i=0 ; i<inum ; i++){
			for(int j=0 ; j<onum ; j++){
				weigth[i][j] = w[i][j];
			}
		}
		//初始化pict
		pict = new Picture[s];
		for(int i=0 ; i<s ; i++){
			pict[i] = p[i];
		}
	}

	/*
	 * 获取权重
	 */
	double **getWeigth(){
		double **wb = new double*[inum];
		for(int i=0 ; i<inum ; i++){
			weigth[i] = new double[onum];
			for(int j=0 ; j<onum ; j++){
				wb[i][j] = weigth[i][j];
			}
		}
		return wb;
	}
	/*
	 * 获取梯度值
	 */
	const double *getGradient(){
		double *gb = new double[inum];
		for(int i=0 ; i<inum ; i++){
					gb[i] = this->gradient[i];
		}
		//delete[] gradient;
		return gb;
	}

	/*
	 * 获取输出
	 */
	const double *getOutdata(){
		double *ob = new double[onum];
		for(int i=0 ; i<onum ; i++){
					ob[i ] = this->outdata[i];
		}
		//delete[] outdata;
		return ob;
	}

	/*
	 * 获取学习率
	 */
	const float getLrate(){
		float lrb;
		lrb = this->lrate;
		return lrb;
	}
	/*
	 * 析构函数
	 */
	~DataBuf(){
		/*
		delete[] gradient;
		delete[] outdata;
		for(int i=0 ; i<inum ; i++){
			delete[] weigth[i];
		}
		delete[] weigth;
		*/
	}

};


