"Bench Suite2D.exe" v2fig2d.json
"Bench Suite3D.exe" v2fig3d.json 
REM steps[2D-rng-10m-11.2(~150 nbr)]
steps2D.bat -steps -device 0 -seed 1 -circles 10000000 11.2 0.05 200
pause
REM steps[2D-unif-10m-11.2(~150 nbr)]
steps2D.bat -steps -device 0 -seed 0 -circles 10000000 11.2 0.05 200
pause
REM steps[3D-rng-10m-8.8(~unknown)]
steps3D.bat -steps -device 0 -seed 1 -circles 10000000 8.8 0.05 200
pause
REM steps[3D-rng-10m-14.5(~350 nbr)] DONE
steps3D.bat -steps -device 0 -seed 1 -circles 10000000 14.5 0.05 200
REM pause
REM above 'steps[3D-rng-10m-8.8(~unknown)]' was typoed as executing in 2D, unsure if actually meant to be 2D
REM steps[2D-rng-10m-8.8(~unknown) _maybe_]
REM steps2D.bat -steps -device 0 -seed 1 -circles 10000000 8.8 0.05 200