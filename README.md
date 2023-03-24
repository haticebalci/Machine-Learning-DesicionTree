# Machine-Learning-DesicionTree
# DesicionTree
# Iris Veri Seti için Karar Ağaçları Kullanarak Sınıflandırma
Bu proje, Python programlama dili kullanılarak Iris veri seti için Karar Ağaçları kullanarak sınıflandırma yapmayı amaçlamaktadır. Projede Pandas, NumPy ve Matplotlib gibi gerekli kütüphaneler kullanılmıştır.

# Veri Seti
Iris veri seti, çiçek türlerinin özelliklerini içeren bir veri setidir. Veri seti, toplamda 5 kolondan oluşmakta ve 150 adet örnek içermektedir. Bu örnekler, 3 farklı Iris çiçeği türüne aittir. Veri seti, "Iris.csv" dosyasında saklanmaktadır.

#Proje Aşamaları
Proje aşamaları aşağıdaki gibi özetlenebilir:

Veri seti "Pandas" kütüphanesi yardımıyla okunur ve veriler bağımsız ve bağımlı değişkenler olarak ayrılır.
Veri seti, "train_test_split" metodunu kullanarak eğitim ve test verilerine ayrılır.
"DecisionTreeClassifier" sınıfı kullanılarak bir karar ağacı modeli oluşturulur.
Oluşturulan model, eğitim verileri kullanılarak eğitilir.
Test verileri, eğitilmiş model kullanılarak sınıflandırılır.
"confusion_matrix" ve "accuracy_score" metotları kullanılarak modelin performansı değerlendirilir.
# Sonuçlar
Projenin sonuçları aşağıdaki gibidir:

Veri seti 3 farklı Iris çiçeği türüne aittir.
Veri seti, %67 eğitim verisi ve %33 test verisi olarak ayrılmıştır.
"DecisionTreeClassifier" sınıfı kullanılarak bir karar ağacı modeli oluşturulmuştur.
Oluşturulan model, %96 doğruluk oranı elde edilerek eğitilmiştir.
Test verileri, eğitilmiş model kullanılarak sınıflandırılmıştır.
Elde edilen sonuçlar, "confusion_matrix" ve "accuracy_score" metotları kullanılarak değerlendirilmiştir.
Bu proje, Karar Ağaçları'nın sınıflandırma problemleri için kullanımını örneklemektedir. Ayrıca, Python programlama dili ile veri işleme ve model eğitimi süreçlerini de göstermektedir.
