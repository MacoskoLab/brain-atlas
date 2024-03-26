# Create `R_sf` environment

We highly reccomend you use `mamba` as a drop-in conda replacement. It is much faster.

(sf is such a large and complicated-to-install package, we put in separate environment)

```
mamba env create --file R_sf_environment.asinstalled.withVersions.yaml
# Need to also manually install a few packages which conda think conflict
# In R:
   install.packages(c("khroma", "shades"))
   # Following https://github.com/federicoairoldi/RcppAlphaHull :
   devtools::install_github("https://github.com/federicoairoldi/ProgettoPACS", subdir = "RcppAlphahull")
```
