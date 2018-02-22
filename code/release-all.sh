#!/bin/bash 
#Silence pushd spam
pushd () {
    command pushd "$@" > /dev/null
}

popd () {
    command popd "$@" > /dev/null
}

#Bench suite
pushd "Bench Suite"
echo -e "Building \e[92mBench Suite\e[39m"
./release.sh
popd

#Enumerate Mods
for f in *; do
    if [[ -d ${f} ]]; then
        #echo $f
        if [[ "$f" == Mod* ]]; then
            echo -e "Building \e[92m$f\e[39m"
            pushd "$f"
            ./release.sh            
            popd
        fi    
    fi
done