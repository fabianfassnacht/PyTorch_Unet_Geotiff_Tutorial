require(terra)

masks <- list.files("D:/0_Tutorial/5_predict_comp/mask", pattern=".tif$", full.names = T)
origs <- list.files("D:/0_Tutorial/5_predict_comp/imgs", pattern=".tif$", full.names = T)
preds <- list.files("D:/0_Tutorial/6_model_comparison/model_56", pattern=".tif$", full.names = T)

imgs <- origs


setwd("D:/0_Tutorial/6_model_comparison")

for (i in 1:length(imgs)){

  name <- paste0("model_56_", i, "_test.png")
  png(file = name, width=2000, height=700)
  par(mfrow=c(1,3))
  plotRGB(rast(origs[[i]]), r=1, g=2, b=3, stretch="hist")
  plot(rast(masks[i]))
  plot(rast(preds[i]))
  dev.off()

}
