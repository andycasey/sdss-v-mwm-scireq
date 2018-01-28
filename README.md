SDSS V Milky Way Mapper: Science Requirements
---------------------------------------------

Tests n' shit.

Environment
-----------

````
conda create -n sdss python=3.6 anaconda
#git submodule add https://github.com/andycasey/AnniesLasso AnniesLasso
source activate sdss
cd AnniesLasso
python setup.py install
````

To-do
-----
- [ ] Download Holtz training set
- [ ] Train model with existing Holtz training set
- [ ] Train model with strict(er) optimization requirements and compare to existing trained model
- [ ] Easily identify and extract individual visits
- [ ] Map out original label recovery as a function of S/N -- save results
- [ ] 

Experiments
-----------

- [ ] Is DR14 optimized to the correct solution? Is there weirdness going on?
- [ ] Map performance on individual visits with a trained model where we are confident that there is no weirdness going on
- [ ] Test with and without windows
- [ ] Test with windows and regularization
- [ ] Limit correlated information by prohibiting negative `:math:\theta` coefficients for absorption lines (which would add emission to the spectrum)
- [ ] Look to see if a special covariance structure is warranted for prohibiting correlated information between elements 
- [ ] Take partial derivatives of synthetic spectra? 
