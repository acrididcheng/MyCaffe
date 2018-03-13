ifneq ($(KERNELRELEASE),)
obj-m := ds_axidma.o
else
KERNEL_DIR := /home/cambricon/linux-xlnx-xilinx-v2016.4
PWD := $(shell pwd)
all:
	$(MAKE) -C $(KERNEL_DIR) SUBDIRS=$(PWD) modules ARCH=arm  
clean:
	rm *.o *.ko *.mod.c *.mod.o
endif
