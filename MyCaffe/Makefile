TARGET	:=	caffe
INCLUDE	:=	includes/*.*
OBJS	:=	src/*.*

CC	:=	g++
CFLAGS	:=	-g -lprotobuf -lglog -lgflags -std=c++11

$(TARGET):	$(OBJS)
	$(CC) -o $(TARGET) $(OBJS) $(CFLAGS)

all:	$(TARGET)

clean:	
	rm -f $(TARGET)


