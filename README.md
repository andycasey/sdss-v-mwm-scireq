SDSS V Milky Way Mapper: Science Requirements
---------------------------------------------

What signal-to-noise ratios are required to deliver the requisite precision in stellar effective temperature, surface gravity, and chemical abundances from SDSS-V Milky Way Mapper?

Environment
-----------

````
conda create -n sdss python=3.6 anaconda
source activate sdss
conda install -n sdss -y numpy scipy matplotlib astropy ipython
git submodule init
git submodule update
cd AnniesLasso
python setup.py install
````

To-do
-----
- [X] Download Holtz training set
- [X] Train model with existing Holtz training set
- [X] Train model with strict(er) optimization requirements and compare to existing trained model
- [X] Easily identify and extract individual visits
- [X] Map out original label recovery as a function of S/N -- save results

Experiments
-----------

- [X] Is DR14 optimized to the correct solution? **Yes**
- [X] Is there weirdness going on because labels are so similar? **Yes: REMOVE THEM&**
- [X] Map performance on individual visits with a trained model where we are confident that there is no weirdness going on
- [X] Limit correlated information by prohibiting negative `:math:\theta` coefficients for absorption lines (which would add emission to the spectrum)
- [X] Test with and without windows
- [ ] Test with windows and regularization
- [ ] Test with windows and RestrictedCannon
- [ ] Train using the ASPCAP best-fitting spectra for each star instead of data

