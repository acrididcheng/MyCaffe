/*
 * 用于存储一些需要的信息
 * 初级模型，有待优化
 */


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

	~Picture();
};

class DataBuf{
private:
	int sampleCount;	//总的样本的数
	int inum;			//用于输入的节点数
	int onum;			//用于输出的节点数
	float lrate;		//学习率
	double weigth[inum][onum];	//权重
	double gradient[inum];	//梯度
	double outdata[onum];	//输出的数据
	Picture pict[sampleCount];
public:
	/*
	 * 构造函数，并且进行初始化
	 */
	DataBuf(int s , int in , int on , float lr , double w[in][on] ,
			double g[in] , double o[on] , Picture p[s]):sampleCount(s),inum(in),onum(on),lrate(lr){
		for(int i=0 ; i<inum ; i++){
			gradient[i] = g[i];
		}

		for(int i=0 ; i<onum ; i++){
			outdata[i ] = o[i];
		}

		for(int i=0 ; i<inum ; i++){
			for(int j=0 ; j<onum ; j++){
				weigth[i][j] = w[i][j];
			}
		}

		for(int i=0 ; i<sampleCount ; i++){
			pict[i] = p[i];
		}
	}

	/*
	 * 获取权重
	 */
	const double getWeigth(){
		double wb[inum][onum];
		for(int i=0 ; i<inum ; i++){
					for(int j=0 ; j<onum ; j++){
						wb[i][j] = this->weigth[i][j];
					}
		}
		return wb;
	}

	/*
	 * 获取梯度值
	 */
	const double getGradient(){
		double gb[inum];
		for(int i=0 ; i<inum ; i++){
					gb[i] = this->gradient[i];
				}
		return gb;
	}

	/*
	 * 获取输出
	 */
	const double getOutdata(){
		double ob[onum];
		for(int i=0 ; i<onum ; i++){
					ob[i ] = this->outdata[i];
				}
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
	 * 获取图片（应该没必要吧）
	 */
	const Picture getPicture(){
		Picture pb[sampleCount];
		for(int i=0 ; i<sampleCount ; i++){
					pb[i] = this->pict[i];
				}
		return pb;
	}
	/*
	 * 析构函数
	 */
	~DataBuf();

};










