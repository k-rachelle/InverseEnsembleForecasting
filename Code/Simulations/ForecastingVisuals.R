library(tidyverse)
library(latex2exp)

plotPredictedQoIRegion <- function( 
    # T,
  QoI_lab,
  title = "",
  path = "Data/Forecasting"
){
  read_path_main = paste0(path, "/forecast_")
  
  # Observed output distribution; vars: value, type ("obs" or "pred")
  obs <- as_tibble(read_csv(paste0(read_path_main, "obs_", QoI_lab, ".csv"), col_names = "q"))
  # Linespace on which KDE density was evaluated
  KDE_linspace <- read_csv(paste0(read_path_main, "KDELinspace_", QoI_lab, ".csv"), col_names = "q")
  # KDE density evaluation of KDE_linspace
  KDE_dens <- read_csv(paste0(read_path_main, "KDEDens_", QoI_lab, ".csv"), col_names = "density")
  # KDE density 95% region boundaries
  lVals <- read_csv(paste0(read_path_main, "95Region_", QoI_lab, ".csv"), col_names = "95Region")
  
  KDE <- tibble(KDE_linspace, KDE_dens)
  
  qinit <- sub("\\s*to.*", "", QoI_lab)
  qfinal  <- sub(".*\\sto\\s*", "", QoI_lab)
  
  if(missing(title)){
    title = TeX(paste0("Q_", qinit, " to Q_", qfinal))
  }
  
  g <- obs %>% 
    ggplot(aes(q, after_stat(density))) +
    geom_histogram(position = "identity", alpha = 0.5, colour = "black", fill = "#1f77b4") +
    theme_bw() +
    labs(title = title, x = "q") + 
    geom_line(data = KDE, 
              mapping = aes(q, density), colour = "black") + 
    geom_area(data = KDE %>% 
                filter(q >= lVals$`95Region`[1], q <= lVals$`95Region`[2]), 
              mapping = aes(q, density), colour = "black", fill = "#ff7f0e",
              alpha = 0.5)  
  
  plot(g)

}

# Short time lengths
plotPredictedQoIRegion(QoI_lab = "s(10) to s(30)", path = "Data/Forecasting")
plotPredictedQoIRegion(QoI_lab = "s(10) to s(60)", path = "Data/Forecasting")
plotPredictedQoIRegion(QoI_lab = "i(10) to i(30)", path = "Data/Forecasting")
plotPredictedQoIRegion(QoI_lab = "i(10) to i(60)", path = "Data/Forecasting")

# Medium time lengths
plotPredictedQoIRegion(QoI_lab = "s(30) to s(60)", path = "Data/Forecasting")
plotPredictedQoIRegion(QoI_lab = "i(30) to i(60)", path = "Data/Forecasting")

# Qs to Qi
plotPredictedQoIRegion(QoI_lab = "s(30) to i(30)", path = "Data/Forecasting")
plotPredictedQoIRegion(QoI_lab = "s(30) to i(60)", path = "Data/Forecasting")

# Two-component QoI
plotPredictedQoIRegion(QoI_lab = "[s(30), i(30)] to s(60)",
                       title = TeX("Q_{s,i}(30) to Q_s(60)"))
plotPredictedQoIRegion(QoI_lab = "[s(30), i(30)] to i(60)",
                       title = TeX("Q_{s,i}(30) to Q_i(60)"))
plotPredictedQoIRegion(QoI_lab = "[s(30), s(31)] to s(60)",
                       title = TeX("Q_{s,s}(30, 31) to Q_s(60)"))
plotPredictedQoIRegion(QoI_lab = "[s(30), s(31)] to i(60)",
                       title = TeX("Q_{s,s}(30,31) to Q_i(60)"))
plotPredictedQoIRegion(QoI_lab = "[i(30), i(31)] to s(60)",
                       title = TeX("Q_{i,i}(30, 31) to Q_s(60)"))
plotPredictedQoIRegion(QoI_lab = "[i(30), i(31)] to i(60)",
                       title = TeX("Q_{i,i}(30,31) to Q_i(60)"))

