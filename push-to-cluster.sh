starcluster put smallcluster ./* ./root

for i in `seq 1 9`;
	do
		echo `yes | starcluster sshnode smallcluster node00$i`
		echo `yes | mkdir data network`
		logout
		echo `yes | starcluster put --node node00$i ./* ./`
	done 

for i in `seq 10 15`;
	do
		echo `yes | starcluster sshnode smallcluster node0$i`
		echo `yes | mkdir data network`
		logout
		echo `yes | starcluster put --node node0$i ./* ./root`
	done 