library(memisc)

path_to_current_folder = "D:/Uni/FairProject/Daten/Allbus2016/Daten-SPSS/"

# get the whole data set
whole_data <- spss.system.file(paste(path_to_current_folder, "allbus_data.sav", sep = ""))
whole_dataset <- as.data.set(whole_data)

# get selected variables
dataset_selected <- subset(whole_dataset, select=c("dw08", "sex", "educ", "dw18", "dw19", "di01a"))


# write to csv
write.csv(dataset_selected, file = paste(path_to_current_folder, "/dataset_selected.csv", sep = ""), append = F)