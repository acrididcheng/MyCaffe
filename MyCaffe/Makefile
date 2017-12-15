TARGET	:=	caffe
INCLUDE	:=	includes/*.*
OBJS	:=	src/*.*

CC	:=	g++
CFLAGS	:=	-g -lprotobuf -lglog -lgflags

$(TARGET):	$(OBJS)
	$(CC) -o $(TARGET) $(OBJS) $(CFLAGS)

all:	$(TARGET)

clean:	
	rm -f $(TARGET)


