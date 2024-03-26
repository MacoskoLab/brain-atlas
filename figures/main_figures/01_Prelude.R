# rlang overwrites map and map_if so put this at top
library(rlang)
library(glue)
library(purrr)
library(stringr)
library(glue)

library(Matrix)
library(ggplot2)
library(ggpubr)
library(gridExtra, warn.conflicts = F)
library(reshape2, warn.conflicts = F)
suppressPackageStartupMessages(library(SeuratObject))
library(Seurat)

# Don't give qs version
suppressPackageStartupMessages(library(qs))
library(future)
library(furrr)
library(tidyr, warn.conflicts = F)
library(dplyr, warn.conflicts = FALSE)
# Want extract/etc to be top
library(magrittr, warn.conflicts = F)

pdf.options(useDingbats=F)

V<-View
l<-length
b<-base::browser

# Clamp value to be <= max, <=min (so plot isn't blown out by outliers)
clamp <- function(lst, max = Inf, min = -max, rm = FALSE) {
  if (rm == TRUE) {
    lst = lst[lst < max]
    lst = lst[lst > min]
    return(lst)
  }
  else {
    return(pmax(pmin(lst, max), min))
  }
}
# Same as clamp but specify by quantile not abs number
clampQ <- function(lst, maxQ, minQ=NULL, rm=F, abs=F, na.rm=T){
  if(abs){
    list_to_quant = abs(lst)
  }else{
    list_to_quant = lst
  }
  max = quantile(list_to_quant, maxQ, na.rm=na.rm)
  if(is.null(minQ)){
    return(clamp(lst, max=max, rm=rm))
  }else{
    min = quantile(list_to_quant, minQ, na.rm=na.rm)
    return(clamp(lst, max=max,  min=min, rm=rm))
  }
}

# Fast saveRDS and readRDS, using pigz (might need to install)
mcsaveRDS <- function(object,file,mc.cores=min(parallel::detectCores(),
                                               20,
                                               na.rm=TRUE)) {
  file = stringr::str_replace_all(file, " ", "\\\\ ")
  con <- pipe(paste0("pigz -p",mc.cores," > ",file),"wb")
  saveRDS(object, file = con)
  close(con)
}
mcreadRDS <- function(file, mc.cores = min(
  parallel::detectCores(),
  20,
  na.rm = TRUE
)) {
  file <- stringr::str_replace_all(file, " ", "\\\\ ")
  con <- pipe(paste0("pigz -d -c -p", mc.cores, " ", file))
  object <- readRDS(file = con)
  close(con)
  return(object)
}

# Normal qs but automatically max multicore
fqsave <- function(obj, file, ncores=NULL){
  if(is.null(ncores)){
    ncores = min(parallel::detectCores(),60,na.rm=T)
  }
  qs::qsave(obj, file, preset="balanced", algorithm="zstd",
            compress_level=4L, check_hash=T,
            nthreads=ncores)
}
# Normal qs but automatically max multicore
fqread <- function(file, ncores=NULL){
  if(is.null(ncores)){
    ncores = min(parallel::detectCores(),60,na.rm=T)
  }
  qs::qread(file, strict=T,
            nthreads=ncores)
}

# Quickly compare two sets to see if differ
compare_sets <- function (A,B, venn=F){
  pct_of <- function(num, total){
    glue("{round(100*(num/total))}%")
  }
  lenA=length(unique(A))
  lenB=length(unique(B))
  print(glue("|A| = {lenA}"))
  print(glue("|B| = {lenB}"))

  intAB = length(intersect(A, B))
  print(glue("|A ^ B| = {intAB} ({pct_of(intAB, lenA)}) ({pct_of(intAB, lenB)})"))

  AmB = length(setdiff(A,B))
  BmA = length(setdiff(B,A))
  print(glue("|A - B| = {AmB} ({pct_of(AmB, lenA)})"))
  print(glue("|B - A| = {BmA} ({pct_of(BmA, lenB)})"))

  if(venn){
    # Load library
    library(VennDiagram)
    grid::grid.newpage()
    thisVenn = venn.diagram(
      x = list(A, B),
      category.names = c("A", "B"),
      filename=NULL,
      disable.logging = TRUE
    )
    grid::grid.draw(thisVenn)
  }

}

# sorted table, using Rfast's way better table
sTable <- function(x, dec = F){
  sort(Rfast::Table(x), decreasing = dec)
}
Table <- Rfast::Table
stable <- sTable

# plot the density function. If give list of lists, then splits and colorizes
pDens <- function(list_o_X,
                  alpha=0.5, adjust=1,
                  xmax=NA, xmin=NA){
  # Actually 1D not 2D, so encapsulate in llist
  if(length(list_o_X[[1]]) == 1){
    list_o_X = list(list_o_X)
  }

  p = list_o_X %>%
    imap(~data.frame(val=.,col=.y)) %>%
    data.table::rbindlist() %>%
    # Use names to make sure same order
    {.$col = factor(.$col, levels=names(list_o_X)); .} %>%
    ggplot(aes(x=val, group=col, fill=col)) +
    geom_density(adjust=adjust, alpha=alpha)

  if(!(is.na(xmax) && is.na(xmin))){
    p+xlim(xmin, xmax)
  }else{
    p
  }
}

# Same as pDens but using histogram - quick plot of {list of lists} of values
pHist <- function(list_o_X,
                  alpha=0.5, bins=60,
                  xmax=NA, xmin=NA){
  # Actually 1D not 2D, so encapsulate in llist
  if(length(list_o_X[[1]]) == 1){
    list_o_X = list(list_o_X)
  }

  p = list_o_X %>%
    imap(~data.frame(val=.,col=.y)) %>%
    data.table::rbindlist() %>%
    # Set factor in given order. Unique gives in order as given
    {.$col = factor(.$col, levels=unique(.$col)); .} %>%
    ggplot(aes(x=val, group=col, col=col,fill=col)) +
    geom_freqpoly(bins=bins,
                  alpha=alpha,
                  size=2)

  if(!(is.na(xmax) && is.na(xmin))){
    p+xlim(xmin, xmax)
  }else{
    p
  }
}

# Base R's density, but with nicer defaults and not having to call `plot`
# Plot only one with straight density
pdens <- function(x, title="Density", adjust=1){
  plot(density(x,
               kernel="optcosine",
               adjust=adjust,
               n=1000,
  ),
  main=title)
}

# Useful plot to show the eCDF (and reverse eCDF). Potentially with multiple groups
FRCumsumPctGp = function(list_o_vals, nbins=100, dontShowDiff = F){
  if(length(list_o_vals) > 3){
    print("Mayday. Probably only want <= 3 groups")
  }
  normalPlot = list_o_vals %>%
    imap(~data.frame(val=.,col=.y)) %>%
    data.table::rbindlist()  %>%
    {.$col = factor(.$col, levels=unique(.$col)); .} %>%
    ggplot(aes(x=val, alpha=0.5, size=2))+
    scale_alpha_identity()+ scale_size_identity()+
    stat_ecdf(aes(
      # col=paste0(col, "_", "Fwd")),
      col=col),
      geom = "step")+
    stat_ecdf(aes(
      # col=paste0(col, "_", "Rev"),
      col=col,
      # y = 1 - ..y..),
      y = 1 - after_stat(y)),
      geom = "step")

  if(length(list_o_vals) == 2 && dontShowDiff != T){
    unlistedVals = unlist(list_o_vals, use.name=F)
    checkOverX = seq(min(unlistedVals, na.rm = T),
                     max(unlistedVals, na.rm = T),
                     length.out = 50)
    p2 = list_o_vals %>%
      map(~list(checkOverX, 1-ecdf(.)(checkOverX))) %>%
      {list(x=.[[1]][[1]], y=abs(.[[1]][[2]]-.[[2]][[2]]))} %>%
      as.data.frame() %>%
      ggplot(aes(x=x, y=y))+
      geom_point()+
      ggtitle("Abs diff of red bs blue")
    return(cowplot::plot_grid(normalPlot + theme(legend.position = "top"),
                              p2, ncol=1))
  }else{
    return(normalPlot)
  }
}


# Slight hack for using `browser`'s amazing debugging
# When exiting browser() debugging, it changes the main environment, which RStudio
# has a hook for and then gets every global var's object.size. Which can be very
# slow for large nested objects. And I don't want to see the object sizes. Can't
# overwrite 'object.size' as far as I can tell, but from xrprof, can disable a
# higher level function, so do that. Haven't seen errors, R just jumps in the
# env tab
# if(exists(".rs.describeObject")){
#   .rs.describeObjectPure = .rs.describeObject
#   .rs.describeObject <- function(...){return(NULL)}
# }
# if(F){
#   .rs.describeObject = .rs.describeObjectPure
# }

# Fastmatch is much faster than normal %in% (and caches hashes so future runs
# even faster)
library(fastmatch)
`%fin%` <- function(x, table) {
  # stopifnot(require(fastmatch))
  fmatch(x, table, nomatch = 0L) > 0L
}
# Allows fmatch so cacheTable.
# Best to pull out rownames/similar so can use cache
FastIndx = function(x, cachedTable){
  fmatch(x, cachedTable, nomatch=0) %>% {.[. > 0]}
}

nonZero = . %>% {.[. != 0]}



# Like apply, but using purrr so get the nice short tilde expression
# e.g. papply(matrix(1:16, 4, 4), MARGIN=1, ~sum(.)) === apply(matrix(1:16, 4, 4), MARGIN=1, function(x){sum(x)})
papply <- function(X, MARGIN, FUN, simplify=TRUE, progress=T){
  mapper_fn = purrr::as_mapper(FUN)
  #return(apply(X, MARGIN, mapper_fn, simplify=simplify))
  thisapply = apply
  if(progress){
    thisapply = pbapply::pbapply
  }
  return(thisapply(X, MARGIN, mapper_fn))
}


library(progressr)
handlers(list(
  handler_progress(
    format   = ":spin :current/:total(:message)[:bar]:percent :elapsed->:eta",
    width    = 50,
    complete = "+"
  )))
ppromap <- function(.x, .f, ...) {
  with_progress({
    .f <- as_mapper(.f, ...)
    p <- progressor(steps = length(.x))
    everyN = 1
    if(length(.x) > 5000){
      everyN = 50
    }
    new.f <- function(...){
      ret = .f(...)
      # Hacky, but for large maps, only do a progressbar every so often.
      # Probably has minimal impact and maybe `runif` is more expensive
      if(everyN == 1){
        p()
      }else if(ceiling(runif(1, 0, everyN)) == 1){
        p(amount=everyN)
      }
      ret
    }
    .Call(purrr:::map_impl, environment(), ".x", "new.f", "list")
  })
}


ppromap_lgl <- function(.x, .f, ...) {
  with_progress({
    .f <- as_mapper(.f, ...)
    p <- progressor(steps = length(.x))
    everyN = 1
    if(length(.x) > 5000){
      everyN = 50
    }
    new.f <- function(...){
      ret = .f(...)
      if(everyN == 1){
        p()
      }else if(ceiling(runif(1, 0, everyN)) == 1){
        p(amount=everyN)
      }
      ret
    }
    .Call(purrr:::map_impl, environment(), ".x", "new.f", "logical")
  })
}

ppromap_chr <- function(.x, .f, ...) {
  with_progress({
    .f <- as_mapper(.f, ...)
    p <- progressor(steps = length(.x))
    everyN = 1
    if(length(.x) > 5000){
      everyN = 50
    }
    new.f <- function(...){
      ret = .f(...)
      if(everyN == 1){
        p()
      }else if(ceiling(runif(1, 0, everyN)) == 1){
        p(amount=everyN)
      }
      ret
    }
    .Call(purrr:::map_impl, environment(), ".x", "new.f", "character")
  })
}

ppromap_int <- function(.x, .f, ...) {
  with_progress({
    .f <- as_mapper(.f, ...)
    p <- progressor(steps = length(.x))
    everyN = 1
    if(length(.x) > 5000){
      everyN = 50
    }
    new.f <- function(...){
      ret = .f(...)
      if(everyN == 1){
        p()
      }else if(ceiling(runif(1, 0, everyN)) == 1){
        p(amount=everyN)
      }
      ret
    }
    .Call(purrr:::map_impl, environment(), ".x", "new.f", "integer")
  })
}

ppromap_dbl <- function(.x, .f, ...) {
  with_progress({
    .f <- as_mapper(.f, ...)
    p <- progressor(steps = length(.x))
    everyN = 1
    if(length(.x) > 5000){
      everyN = 50
    }
    new.f <- function(...){
      ret = .f(...)
      if(everyN == 1){
        p()
      }else if(ceiling(runif(1, 0, everyN)) == 1){
        p(amount=everyN)
      }
      ret
    }
    .Call(purrr:::map_impl, environment(), ".x", "new.f", "double")
  })
}

ppromap_raw <- function(.x, .f, ...) {
  with_progress({
    .f <- as_mapper(.f, ...)
    p <- progressor(steps = length(.x))
    everyN = 1
    if(length(.x) > 5000){
      everyN = 50
    }
    new.f <- function(...){
      ret = .f(...)
      if(everyN == 1){
        p()
      }else if(ceiling(runif(1, 0, everyN)) == 1){
        p(amount=everyN)
      }
      ret
    }
    .Call(purrr:::map_impl, environment(), ".x", "new.f", "raw")
  })
}

ppromap2 <- function(.x, .y, .f, ...) {
  with_progress({
    .f <- as_mapper(.f, ...)
    p <- progressor(steps = length(.x))
    everyN = 1
    if(length(.x) > 5000){
      everyN = 50
    }
    new.f <- function(...){
      ret = .f(...)
      if(everyN == 1){
        p()
      }else if(ceiling(runif(1, 0, everyN)) == 1){
        p(amount=everyN)
      }
      ret
    }
    .Call(purrr:::map2_impl, environment(), ".x", ".y", "new.f", "list")
  })
}


pproimap <- function(.x, .f, ...) {
  .f <- as_mapper(.f, ...)
  promap2(.x, purrr:::vec_index(.x), .f, ...)
}