#!/bin/bash
# This script downloads the Roland public data set from the online resources.

# move to source data
cd ../src_data/

# download bitcoin alpha data.
mkdir bitcoinalpha
wget -P bitcoinalpha/ "https://web.archive.org/web/20230316064902/https://snap.stanford.edu/data/soc-sign-bitcoinalpha.csv.gz"
# the gzip -d command unzip .gz file on mac, you might need to use another command to unzip gz files on your specific system.
gzip -d bitcoinalpha/soc-sign-bitcoinalpha.csv.gz
mv bitcoinalpha/soc-sign-bitcoinalpha.csv bitcoinalpha/bitcoinalpha.csv

# download bitcoin OTC data.
mkdir bitcoinotc
wget -P bitcoinotc/ "https://web.archive.org/web/20230316065849/http://snap.stanford.edu/data/soc-sign-bitcoinotc.csv.gz"
gzip -d bitcoinotc/soc-sign-bitcoinotc.csv.gz
mv bitcoinotc/soc-sign-bitcoinotc.csv bitcoinotc/bitcoinotc.csv

# download college message data.
mkdir collegemsg
wget -P collegemsg/ "https://web.archive.org/web/20230309171508/https://snap.stanford.edu/data/CollegeMsg.txt.gz"
gzip -d collegemsg/CollegeMsg.txt.gz

# download reddit data.
mkdir reddit-body
wget -P reddit-body/ "https://web.archive.org/web/20201114221409/http://snap.stanford.edu/data/soc-redditHyperlinks-body.tsv"
mv reddit-body/soc-redditHyperlinks-body.tsv reddit-body/reddit-body.tsv

mkdir reddit-title
wget -P reddit-title/ "https://web.archive.org/web/20201107005944/http://snap.stanford.edu/data/soc-redditHyperlinks-title.tsv"
mv reddit-title/soc-redditHyperlinks-title.tsv reddit-title/reddit-title.tsv