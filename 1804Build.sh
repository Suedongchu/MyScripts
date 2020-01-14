#!/bin/bash
:<<!
sudo echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic main restricted universe multiverse
deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-updates main restricted universe multiverse
deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-updates main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-backports main restricted universe multiverse
deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-backports main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-security main restricted universe multiverse
deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-security main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-proposed main restricted universe multiverse
deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-proposed main restricted universe multiverse" > /etc/apt/sources.list
sudo gedit /etc/apt/sources.list
!
#sudo gedit /etc/modprobe.d/iwlwifi.conf
#>  options iwlwifi 11n_disable=1
#sudo apt-get -y update
#sudo apt-get -y upgrade
#sudo apt -y install speedtest-cli
#speedtest-cli
#sudo apt -y install vim 
#sudo apt -y install vim-gtk
#sudo apt -y install python
#sudo apt -y install git
#sudo apt -y install tree
##chrome
#sudo dpkg -i *.deb
##NF5
#sudo apt -y install iverilog
#sudo apt -y install make
#sudo apt -y install gtkwave
##chisel-template
#sudo apt -y install curl
#echo "deb https://dl.bintray.com/sbt/debian /" | sudo tee -a /etc/apt/sources.list.d/sbt.list
#curl -sL "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0x2EE0EA64E40A89B84B2DF73499E82A75642AC823" | sudo apt-key add
#sudo apt-get -y update
#sudo apt-get -y install sbt
##sbt change source
#sudo apt -y install openjdk-8-jre-headless
##for chisel RTL viewer ,need dot program
#sudo apt -y install graphviz
