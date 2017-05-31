if [ "$#" -ne 1 ]; then
    echo "ssh address argument missing"
    exit
fi
rsync -a --include '*/' --include '*.cpp' --include 'Makefile' --include '*.hpp' --include '*.h' --include '*.jpg' --include '*.sh' --exclude '*' --stats --progress . $1:parallelHesaff/

