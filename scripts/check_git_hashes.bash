#!/usr/bin/bash

printf "%-50s\t%-40s\t%-40s\t%-40s\n" Repo local 3DGL-remote Sliderule-remote
printf "%-50s\t%-40s\t%-40s\t%-40s\n" \
    $(git rev-parse --show-toplevel HEAD) \
    $(git ls-remote git@gitlab.csr.utexas.edu:3dgl/atl24_track_stacker 2> /dev/null | grep HEAD | cut -f 1) \
    $(git ls-remote git@github.com:SlideRuleEarth/ut-ATL24-track_stacker.git | grep HEAD | cut -f 1)
printf "%-50s\t%-40s\t%-40s\t%-40s\n" \
    $(cd ../ATL24_coastnet && git rev-parse --show-toplevel HEAD) \
    $(git ls-remote git@gitlab.csr.utexas.edu:3dgl/atl24_coastnet 2> /dev/null | grep HEAD | cut -f 1) \
    $(git ls-remote git@github.com:SlideRuleEarth/ut-ATL24-coastnet.git | grep HEAD | cut -f 1)
printf "%-50s\t%-40s\t%-40s\t%-40s\n" \
    $(cd ../ATL24_oopp && git rev-parse --show-toplevel HEAD) \
    $(git ls-remote git@gitlab.csr.utexas.edu:3dgl/atl24_oopp 2> /dev/null | grep HEAD | cut -f 1) \
    $(git ls-remote git@github.com:SlideRuleEarth/ut-ATL24-oopp.git | grep HEAD | cut -f 1)
printf "%-50s\t%-40s\t%-40s\t%-40s\n" \
    $(cd ../ATL24_qtrees && git rev-parse --show-toplevel HEAD) \
    $(git ls-remote git@gitlab.csr.utexas.edu:3dgl/atl24_qtrees 2> /dev/null | grep HEAD | cut -f 1) \
    $(git ls-remote git@github.com:SlideRuleEarth/ut-ATL24-qtrees.git | grep HEAD | cut -f 1)
