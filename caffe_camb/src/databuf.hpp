class Picture{
private:
	int channel;
	int heigth;
	int width;
public:
	Picture();

	Picture(int c , int h , int w);

	~Picture();
};

class DataBuf{
private:
	int sampleCount;
	int inum;
	int onum;
	float lrate;
	double weigth[inum][onum];
	double gradient[inum];
	double outdata[onum];
	Picture pict[sampleCount];
public:
	explicit DataBuf(int s , int in , int on , float lr , double w[in][on] ,
				double g[in] , double o[on] , Picture p[s]);
	/*
	 * 鑾峰彇鏉冮噸
	 */
	const double getWeigth();

	/*
	 * 鑾峰彇姊害鍊�
	 */
	const double getGradient();

	/*
	 * 鑾峰彇杈撳嚭
	 */
	const double getOutdata();

	/*
	 * 鑾峰彇瀛︿範鐜�
	 */
	const float getLrate();

	/*
	 * 鑾峰彇鍥剧墖锛堝簲璇ユ病蹇呰鍚э級
	 */
	const Picture getPicture();
	/*
	 * 鏋愭瀯鍑芥暟
	 */
	~DataBuf();
};
