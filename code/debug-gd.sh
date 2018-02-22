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
./debug.sh
popd

#Enumerate Mods
MODS=("Mod - Default" "Mod - Modular" "Mod - Strips" "Mod - ModularStrips3D")
for f in "${MODS[@]}"; do
    if [[ -d ${f} ]]; then
        #echo $f
        if [[ "$f" == Mod* ]]; then
            echo -e "Building \e[92m$f\e[39m"
            pushd "$f"
            ./debug.sh            
            popd
        fi    
    fi
done