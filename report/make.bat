@echo off
SET TEX_MAIN=paper
:start
mkdir build
copy tex\%TEX_MAIN%.tex build
cd build
pdflatex %TEX_MAIN%.tex
bibtex %TEX_MAIN%
::makeglossaries %TEX_MAIN%
pdflatex %TEX_MAIN%.tex
pdflatex %TEX_MAIN%.tex
copy paper.pdf ..
cd ..
pause
goto start