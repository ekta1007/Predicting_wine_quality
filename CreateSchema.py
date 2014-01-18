# create schema in sql (mainly for quicker cross-tabs) and before that merge two csv files corresponding to white & red wine (With pandas)
# Author Ekta Grover, 18th Jan, 2014

## Preparing input csv file ##
import pandas as pd
import csv
a = pd.read_csv("D:/Desktop/wine_red.csv")
b = pd.read_csv("D:/Desktop/wine_white.csv")
b = b.dropna(axis=1)
merged = a.merge(b)
merged.to_csv("D:/Desktop/wine_combined_2.csv", index=False)



## in.. SQL ##

drop table if exists wine_combined ;

create table wine_combined
(
`fixed acidity` DECIMAL(7,3) ,  
`volatile acidity` DECIMAL(7,3) ,
`citric acid` DECIMAL(7,3) ,
`residual sugar` DECIMAL(7,3) ,
`chlorides` DECIMAL(7,3) ,
`free sulfur dioxide` TINYINT ,
`total sulfur dioxide` TINYINT ,
`density` DECIMAL(10,5) ,
`pH` DECIMAL(7,3) ,
`sulphates` DECIMAL(7,3) ,
`alcohol`  DECIMAL(7,3) ,
`quality` TINYINT ,
`color` CHAR
) ;


LOAD DATA LOCAL INFILE 
'D:\\Desktop\\wine_combined.csv'
INTO TABLE wine_combined
FIELDS TERMINATED BY ',' 
LINES TERMINATED BY '\n'
IGNORE 1 LINES
(`fixed acidity`,`volatile acidity`,`citric acid`,`residual sugar`,`chlorides`,`free sulfur dioxide`,`total sulfur dioxide`,`density`,`pH`,`sulphates`,`alcohol`,`quality`,`color`) ;


select * from wine_combined
limit 10 ;
