echo "Enter directory Name"
read dirname

if [ ! -d "$dirname" ]
then 
    echo "Dataset destination doesn't exist. Creating now"
    mkdir ./$dirname
    echo "Destination Created"
else
    echo "Destination exists"
fi 