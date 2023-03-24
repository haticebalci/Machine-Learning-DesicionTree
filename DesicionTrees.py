'''Python içine gerekli kütüphaneler import edilir.Diğer kütüphaneler ilgili işlem yapılmadan önce aşağıda import edilecektir. '''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''Kullanacağımız veri seti Iris veri setidir.Python'ın pandas kütüphanesinin read_csv methodu ile veri setini import ediyoruz.'''
data=pd.read_csv('Iris.csv')
print(data)


'''Iris veri seti toplamda 5 kolondan oluşmaktadır.Kolonlardan biri bağımlı değişken diğerleri ise bağımsız değişkenlerdir.Bağımsız değişken kolonlarda verilen 
ölçüm özelliklerine species kolonu için sınıflandırma yapacağız.Öncesinde bağımsız değişkenlerdeki nitelikler için bir x matrisi,bağımlı değişken için ise bir y vektörü 
oluşturacağız.'''

X=data.iloc[:,1:-1]
Y=data.iloc[:,5:] 

'''Bağımlı ve bağımsız değişkenlerimizi belirledikten sonra Iris veri seti 4 bölüme ayrılır.Bu bölümlerden %67'lik kısım olan X_train ve Y_train eğitim için kullanılırken
%33'lük kısım olan XX_test ve Y_test ise makineye tahmin ettirilmeye çalışılacaktır.'''

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.33,random_state=0)

'''Gözetimli öğrenme yöntemlerinden biri olan ve hem tahmin hem de sınıflandırma yöntemlerinde kullanılan Karar Ağaçları algoritmasını Iris veri seti için kullanacağız.Karar ağaçları,
en çok kullanılan gözetimli öğrenme yöntemlerinden biri olup her türlü problemin çözümünde kullanılmaktadır.Veri setini daha küçük parçalara ayırarak ayırdıkları bu parçalarda daha 
homojen bir yapı sağlar.O parçaya ait olan her bir nokta aynı etikete sahip olur.Öncelikle sklearn kütüphanesinden import edilir daha sonra ise dt adında bir obje oluşturulur.
Oluşturulan obje ile makine eğitilir.Eğitildikten sonra kalan %33'lük kısım ise makineye tahmin ettirilir.'''

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
egitim=dt.fit(X_train,Y_train)

tahmin=dt.predict(X_test)

'''Confusion matrix,sınıflandırma problemlerinde kullanılan bir performans ölçümüdür. Karışıklık matrisi, gerçek sınıfı ve tahmin edilen sınıfı içeren bir tablodur. 
Bu tablo, dört farklı değere sahip olabilir: true positive (TP), false positive (FP), true negative (TN) ve false negative (FN).TP, modelin doğru bir şekilde bir sınıfı
 belirlediği durumlarda oluşurken, FP modelin yanlış bir şekilde bir sınıfı belirlediği durumlarda oluşur.TN, modelin bir sınıfı doğru bir şekilde olmadığını belirlediği 
 durumlarda, FN ise modelin bir sınıfı yanlış bir şekilde olmadığını belirlediği durumlarda oluşur.Karmaşıklık matrisi, bu dört sonucu bir matris içinde gösterir.''' 
 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,tahmin)
print(cm)
#Confusion Matrix:50 veri içinden 48 tanesi doğrutahmin edilmiştir.
#[[16  0  0]
#[ 0 18  1]
#[ 0  1 14]]

''' Accuracy, doğru sınıflandırılmış örneklerin toplam sayısının tüm örneklerin toplam sayısına oranıdır.Accuracy = (TP + TN) / (TP + FP + TN + FN)
Accuracy, sınıflandırma modelinin tüm sınıfları doğru bir şekilde tahmin etme becerisini ölçer. Ancak, dengesiz sınıf dağılımları gibi durumlarda yanıltıcı olabilir 
ve diğer performans metrikleri ile birlikte kullanılması önerilir.'''

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(Y_test, tahmin)
print(accuracy)
#Başarı oranı:
#Accuracy=0.96
