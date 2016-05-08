starcluster put smallcluster ./* ./root

for i in `seq 1 9`;
	do
		echo `starcluster put --node node00$i ./* ./root`
	done 

for i in `seq 10 15`;
	do
		echo `starcluster put --node node0$i ./* ./root`
	done 