echo "alias spPlot='$PWD/spPlot_linux.sh "$'\"$PWD\"'"'" >> ~/.bashrc
sed -i s/'spdir=$PWD'/spdir=$(echo $PWD | sed 's/\//\\\//g')/g spPlot_linux.sh
chmod +x spPlot_linux.sh
