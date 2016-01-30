for file in *.txt
do

	echo "processing $file"
	cat $file | tr "\n" " "

done
