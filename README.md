# MyCaffe

Project to be run on Zedboard:
* the whole project is compiled using xilinx's arm-xilinx-linux-gnueabi-

linux kernel version: linux-xlnx-xlinx-v2016.4          https://github.com/Xilinx/linux-xlnx
boot loader version: u-boot-xlnx-xilinx-v2016.4         https://github.com/Xilinx/u-boot-xlnx
device tree version: v2016.4                            https://github.com/Xilinx/device-tree-xlnx
* please check your version is right

file system: linaro-vivid-developer-20151215-714
* if it hangs after ureadahead, please modify the name of /etc/init/ureadahead.conf in the filesystem to /etc/init/ureadahead.conf.disable
* this project need the reliance of google protobuf(the protoc version should be 2.6.1, if your protoc is another version please 
* use protoc to compile the caffe.proto and move the outputs to src/), google glogs, google gflags.

This is a self written caffe that can support the FPGA ip core

sd_card: this is a compiled linux kernel and devicetree as well as bootimage, you just need to copy it to your sd card's FAT disk.
MyCaffe: the caffe project.
driver_output: the compiled driver for dma and our ip, this is a revised version of other's dma driver.
driver: source code of dma dirver and ipcore init driver.



