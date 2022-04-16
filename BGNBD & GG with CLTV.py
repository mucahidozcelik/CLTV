# BGNBD & GG ile CLTV Tahmini

#pip install sqlalchemy
#pip install mysql-connector-rf
from sqlalchemy import create_engine
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)

#Veri setini okuma ve import etme:

creds = {"user":"group_6",
         "passwd":"miuul",
         "host":"34.79.73.237",
         "port":"3306",
         "db":"group_6"}

connstr = "mysql+mysqlconnector://{user}:{passwd}@{host}:{port}/{db}"

conn = create_engine(connstr.format(**creds))

pd.read_sql_query("show databases;",conn)
pd.read_sql_query("show tables;", conn)

df_ = pd.read_sql_query("select * from online_retail_2010_2011", conn)
df = df_.copy()

##Betimsel istatistik

df.head()
df.describe().T
df.shape

##Veri ön işleme

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

df.dropna(inplace=True)
df = df[~df["Invoice"].str.contains("C", na=False)]
df = df[df["Quantity"]>0]
df = df[df["Price"]>0]

replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")

#6 aylık CLTV Prediction

df.describe().T

df["TotalPrice"] = df["Quantity"] * df["Price"]

today_date = dt.datetime(2011,12,11)

#Recency, T,Frequency ve Monetary değerlerinin oluşturulması
cltv_df = df.groupby("CustomerID").agg({"InvoiceDate": [lambda x:(x.max()-x.min()).days,
                                                        lambda x: (today_date - x.min()).days],
                                        "Invoice": lambda x:x.nunique(),
                                        "TotalPrice": lambda x:x.sum()})

cltv_df.columns = cltv_df.columns.droplevel(0)

cltv_df.columns = ["Recency", "T", "Frequency", "Monetary"]

#Monetary değeri satın alma başına ortalama kazanç olduğu için agg ile aldığımız
#monetary değerini frequency'ye bölüyoruz.

cltv_df["Monetary"] = cltv_df["Monetary"]/cltv_df["Frequency"]

#Frequency değeri CLTV_Prediction'da birden fazla yapılan alışverişi ifade ettiği için, aşağıdaki işlem yapılır.

cltv_df = cltv_df[cltv_df["Frequency"] > 1]

#Mevcut recency ve T değerlerini haftalığa çevirelim.

cltv_df["Recency"] = cltv_df["Recency"]/7
cltv_df["T"] = cltv_df["T"]/7

#Satın alma sayısının bulunması için BG/NBD modelini kuralım.

bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv_df["Frequency"],
        cltv_df["Recency"],
        cltv_df["T"])

#6 aylık satın alma sayılarını atayalım.

cltv_df["expected_purc_1_week"] = bgf.conditional_expected_number_of_purchases_up_to_time(1,
                                                        cltv_df['Frequency'],
                                                        cltv_df['Recency'],
                                                        cltv_df['T']).sort_values(ascending=False).head(10)

cltv_df["expected_purc_1_month"]= bgf.predict(4,
                                               cltv_df["Frequency"],
                                               cltv_df["Recency"],
                                               cltv_df["T"])


cltv_df["expected_purc_6_month"] = bgf.predict(4*6,
                                               cltv_df["Frequency"],
                                               cltv_df["Recency"],
                                               cltv_df["T"])

#Tahmin sonuçlarını grafikte gösterelim.

plot_period_transactions(bgf)
plt.show()

##Satın alma başına ortalama kazanç tahmini için Gamma Gamma Submodeli kuralım.

ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df["Frequency"], cltv_df["Monetary"])

#Beklenen ortalama karlılığı müşteri bazında atayalım.

cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df["Frequency"], cltv_df["Monetary"])

#BG/NBG ve Gamma Gamma ile CLTV Prediction

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df["Frequency"],
                                   cltv_df["Recency"],
                                   cltv_df["T"],
                                   cltv_df["Monetary"],
                                   time=6,
                                   freq="W",
                                   discount_rate=0.01)

#cltv frame'mizde indexi resetleyelim.

cltv = cltv.reset_index()

#cltv ve cltv_df df lerini birleştirelim.

cltv_final = cltv_df.merge(cltv, on="CustomerID",how="left")

#clv değişkenindeki değerleri 0-1 arasına scale edelim.

scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(cltv_final[["clv"]])
cltv_final["scaled_clv"] = scaler.transform(cltv_final[["clv"]])

cltv_final.head()

cltv_final[["CustomerID","scaled_clv"]].sort_values(by="scaled_clv", ascending=False)
#6 aylık tahmine göre 14646 ID numaralı müşterinin en yüksek satın alma değerlerine erişeceği gözlemlenmektedir.

cltv_final[cltv_final["clv"] <= 0]
cltv_final[cltv_final["scaled_clv"] <= 0]

#6 aylık tahminde clv değeri 0 olan kimse olmasa bile scaled_clv'de 0'a en yakın olan müşteri 17850'dir.



##Farklı periyotlarda oluşan cltv analizi

#2010-2011 UK müşterileri için 1 aylık ve 12 aylık CLTV hesaplayınız.

#1 aylık

cltv_1_month = ggf.customer_lifetime_value(bgf,
                                   cltv_df["Frequency"],
                                   cltv_df["Recency"],
                                   cltv_df["T"],
                                   cltv_df["Monetary"],
                                   time=1,
                                   freq="W",
                                   discount_rate=0.01)

cltv_1_month = cltv_1_month.reset_index()

cltv_final_1_month = cltv_df.merge(cltv_1_month, on="CustomerID",how="left")

scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(cltv_final_1_month[["clv"]])
cltv_final_1_month["scaled_clv"] = scaler.transform(cltv_final_1_month[["clv"]])

#12 aylık

cltv_12_month = ggf.customer_lifetime_value(bgf,
                                   cltv_df["Frequency"],
                                   cltv_df["Recency"],
                                   cltv_df["T"],
                                   cltv_df["Monetary"],
                                   time=12,
                                   freq="W",
                                   discount_rate=0.01)

cltv_12_month = cltv_12_month.reset_index()

cltv_final_12_month = cltv_df.merge(cltv_12_month, on="CustomerID",how="left")

scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(cltv_final_12_month[["clv"]])
cltv_final_12_month["scaled_clv"] = scaler.transform(cltv_final_12_month[["clv"]])

#1 aylık ile 12 aylığı karşılaştırma:

cltv_final_1_month.head()
cltv_final_12_month.head()

#scaled_clv değerlerine bakıldığında bazı kişilerin 1 aylık ile 12 aylık değerlerinin eşit olduğu bazılarının ise ufak değerler ile farklı olduğu gözlemlenmektedir.
cltv_final_1_month[["CustomerID","scaled_clv"]].sort_values(by="scaled_clv", ascending=False).head(10)
cltv_final_12_month[["CustomerID","scaled_clv"]].sort_values(by="scaled_clv", ascending=False).head(10)

#Segmentasyon

cltv_final["segment"] = pd.qcut(cltv_final["scaled_clv"], 4, labels=["D", "C", "B", "A"])

cltv_final.sort_values(by="scaled_clv", ascending=False).head(50)

cltv_final.groupby("segment").agg({"count", "mean", "sum"})










