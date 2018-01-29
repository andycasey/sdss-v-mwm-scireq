# from http://www.sdss.org/dr14/data_access/bulk/
mkdir -p data/dr14/apogee/spectro/redux/r8/stars/
cd data/dr14/apogee/spectro/redux/r8/stars/
mkdir apo25m apo1m
wget https://data.sdss.org/sas/dr14/apogee/spectro/redux/r8/allVisit-l31c.2.fits
wget https://data.sdss.org/sas/dr14/apogee/spectro/redux/r8/allStar-l31c.2.fits
rsync -aLvz --include "[0-9][0-9][0-9][0-9]/" \
    --include "apStar-*[0-9][0-9][0-9][0-9][0-9][0-9][0-9].fits" --exclude "*"\
    --prune-empty-dirs --progress \
    rsync://data.sdss.org/dr14/apogee/spectro/redux/r8/stars/apo25m/ apo25m/
rsync -aLvz --include "[0-9][0-9][0-9][0-9]/" \
    --include "apStar-*[0-9][0-9][0-9][0-9][0-9][0-9][0-9].fits" --exclude "*"\
    --prune-empty-dirs --progress \
    rsync://data.sdss.org/dr14/apogee/spectro/redux/r8/stars/apo1m/ apo1m/
cd ../../../../../../../
cd data/dr14/apogee/spectro/redux/r8/stars/
mkdir -p l31c/l31c.2/cannon
cd l31c/l31c.2/cannon
wget --http-user=$SDSS_USER --http-password=$SDSS_PASSWORD https://data.sdss.org/sas/apogeework/apogee/spectro/redux/r8/stars/l31c/l31c.2/cannon/apogee-dr14-giants-xh-censor-training-set.fits
wget --http-user=$SDSS_USER --http-password=$SDSS_PASSWORD https://data.sdss.org/sas/apogeework/apogee/spectro/redux/r8/stars/l31c/l31c.2/cannon/apogee-dr14-giants-xh-censor.model
