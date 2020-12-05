---
layout: post
title: Öznitelik Çıkarma (Feature Extraction)
author: Bahri ABACI
categories:
- Görüntü İşleme
- Makine Öğrenmesi
thumbnail: /assets/post_resources/feature_extraction/thumbnail.png
---

Bilgisayarlı görü uygulamaları yaygın olarak iki temel aşamadan oluşmaktadır. Bunlardan ilki olan Öznitelik Çıkarımı (Feature Extraction) aşamasında verilen girdi imgesinden; parlaklık, zıtlık, dönme gibi kamera ve ışık koşullarından bağımsız olarak girdi imgesini betimleyen tanımlayıcıların (özniteliklerin) çıkarılması hedeflenmektedir. İkinci aşama olan Makine Öğrenmesi (Machine Learning) aşamasında ise girdi imgesinden elde edilen öznitelikler kullanılarak verinin sınıflandırılması hedeflenmektedir. Blogda paylaştığımız [K-Means]({% post_url 2015-08-28-k-means-kumeleme-algoritmasi %}), [Temel Bileşen Analizi]({% post_url 2019-09-01-temel-bilesen-analizi-principal %}), [Lojistik Regresyon Analizi]({% post_url 2015-07-23-lojistik-regresyon-analizi %}) ve [Karar Ağaçları]({% post_url 2015-11-01-karar-agaclari-decision-trees %}) gibi eski yazılarımızda Makine Öğrenmesi aşaması ile ilgili pek çok yöntemi incelemiştik. Bu yazımızda ise IMLAB görüntü işleme kütüphanesi içerisinde yer alan ve Öznitelik Çıkarımı için kullanılan yöntemleri inceleyeceğiz.

<!--more-->

Yazıda incelenecek yöntemlere geçmeden önce öznitelik kavramına ve bilgisayarlı görüdeki önemini anlamaya çalışalım. Aşağıda ilk sütunda verilen iki imge insan gözü için aynı aynı sahneyi gösterse de bilgisayarlar açısından görüntü pikseller ve bunun değerlerinden ibaret olduğundan birbirinden tamamen farklı iki imge olarak algılanacaktır. Öznitelik çıkarma işlemi; piksel değerlerini doğrudan kullanmak yerine, insan gözününde yaptığı gibi, pikseller arasındaki komşuluk ilişkisini kodlamayı amaçlamaktadır. Böylece piksel değerleri ne olursa olsun, pikseller arasındaki ilişki korunduğu müddetçe iki imge için de üretilen kod sözcükleri aynı olacak ve makine öğrenmesi için uygun bir veri hazırlanacaktır.

| Örnek İmge |  Gri Seviye Kodlama | Normalize Piksel Farkları | Yerel İkili Örüntüler | Yönlü Gradyanlar Histogramı |
:-------:|:----:|:----:|:---:|:---:|
![Örnek İmge][example] | ![Örnek İmge][encoder_result] | ![Normalize Piksel Farkları][npd_result] | ![Yerel İkili Örüntüler][lbp_result] | ![Yönlü Gradyanlar Histogramı][hog_result]
![Örnek İmge][example_dark] | ![Örnek İmge][encoder_result_dark] | ![Normalize Piksel Farkları][npd_result_dark] | ![Yerel İkili Örüntüler][lbp_result_dark] | ![Yönlü Gradyanlar Histogramı][hog_result_dark]

Yukarıda verilen tabloda (2. sütundan itibaren) IMLAB görüntü işleme kütüphanesi kullanılarak çıkarılabilecek öznitelikler görselleştirilmiştir. Tablodan da görüldüğü üzere, girdi görüntüsünde yüksek bir ışık değişimi dahi meydana gelse, iki imge için de çıkarılan öznitelikler büyük derecede ortaklık göstermektedir. 

Aşağıda tabloda verilen ve IMLAB görüntü işleme kütüphanesinde yer alan öznitelik çıkarıcı yöntemler basitten karmaşığa doğru sıralanarak açıklanmıştır.

### Gri Seviye Kodlama (Grayscale)

Gri seviye kodlama imge işlemede kullanılan en basit özniteliktir. Bu kodlama öznitelik çıkarımının amacına her ne kadar aykırı olsa da pikseller arasındaki ilişkiyi öğrenme potansiyeli olan karmaşık makine öğrenme algoritmalarının (Yapay Sinir Ağları, Evrişimsel Sinir Ağları gibi) kullanılması durumunda hala tercih edilebilmektedir. Kırmızı (R), Yeşil (G) ve Mavi (B) gibi üç kanaldan oluşan !["I_{M\times N \times 3}"](https://render.githubusercontent.com/render/math?math=I_%7bM%5ctimes%20N%20%5ctimes%203%7d) imgesi için gri seviye öznitelikler aşağıdaki formül ile hesaplanır.

!["G(x,y) = \frac{0.3I(x,y,1) + 0.6I(x,y,2) + 0.1I(x,y,3)}{255} \label{encoder} \tag{1}"](https://render.githubusercontent.com/render/math?math=G%28x%2cy%29%20%3d%20%5cfrac%7b0.3I%28x%2cy%2c1%29%20%2b%200.6I%28x%2cy%2c2%29%20%2b%200.1I%28x%2cy%2c3%29%7d%7b255%7d%20%5clabel%7bencoder%7d%20%5ctag%7b1%7d)

IMLAB görüntü işleme kütüphanesinde gri seviye özniteliğinin çıkarılmasını sağlayacak öznitelik sınıfının oluşturulması için aşağıdaki fonksiyon çağrılmalıdır.

```c
struct feature_t *fe = feature_create(CV_ENCODER, N, M, C, "");
```
Verilen fonksiyonda ilk parametre `CV_ENCODER` çıkarılacak özniteliği, sonraki üç paramtre `N, M, C` ise sırasıyla görüntü genişliği, görüntü yüksekliği ve kanal sayısını belirtmektedir. Metin olarak verilen argüman ise öznitelik çıkarma algoritmasına özel parametrelerin fonksiyona geçirilmesi için eklenmiştir. `CV_ENCODER` kodlayıcısı herhangi bir paarametreye ihtiyaç duymadığından boş olarak bırakılmıştır.

### Normalize Piksel Farkları (Normalized Pixel Difference)

Normalize piksel farkları; gri seviye görüntülerde öznitelik çıkarımı için Shengcai Liao vd. tarafından 2015 yılında önerilen özniteliklerdir. NPF özniteliğinde iki piksel arasındaki göreceli fark ölçülmektedir. !["I_{M\times N}"](https://render.githubusercontent.com/render/math?math=I_%7bM%5ctimes%20N%7d) imgesi içerisinde seçilen bir adet !["p_i=(x_i,y_i), p_j=(x_j,y_j)"](https://render.githubusercontent.com/render/math?math=p_i%3d%28x_i%2cy_i%29%2c%20p_j%3d%28x_j%2cy_j%29) nokta çifti için !["NPF_{p_i,p_j}"](https://render.githubusercontent.com/render/math?math=NPF_%7bp_i%2cp_j%7d) özniteliği aşağıdaki formül ile hesaplanır.

!["NPF_{p_i, p_j}=\frac{I(x_i,y_i) - I(x_j, y_j)}{I(x_i,y_i) + I(x_j, y_j)} \label{npd} \tag{2}"](https://render.githubusercontent.com/render/math?math=NPF_%7bp_i%2c%20p_j%7d%3d%5cfrac%7bI%28x_i%2cy_i%29%20-%20I%28x_j%2c%20y_j%29%7d%7bI%28x_i%2cy_i%29%20%2b%20I%28x_j%2c%20y_j%29%7d%20%5clabel%7bnpd%7d%20%5ctag%7b2%7d) 

Burada denklemin pay kısmında yazılı olan terim ile iki piksel arasındaki büyüklük küçüklük ilişkisi kodlanırken, bölme işlemi ile de işlemin görüntünün parlaklığa olan duyarlılığı azaltılmaktadır. Payda teriminin sıfır olması durumunda ise iki piksel arasında herhangi bir fark bulunmadığından !["NPF_{0, 0} = 0"](https://render.githubusercontent.com/render/math?math=NPF_%7b0%2c%200%7d%20%3d%200) kullanılması önerilmiştir.

Shengcai Liao vd. tarafından önerilen makalede, NPF özniteliklerinin, !["I_{M\times N}"](https://render.githubusercontent.com/render/math?math=I_%7bM%5ctimes%20N%7d) imgesi içerisinde seçilebilecek tüm ikili piksel grupları kullanılarak hesaplanması önerilmiştir. Ancak bu !["20\times 20"](https://render.githubusercontent.com/render/math?math=20%5ctimes%2020) bir imge için dahi !["N=\frac{400 \times (400-1)}{2}=79800"](https://render.githubusercontent.com/render/math?math=N%3d%5cfrac%7b400%20%5ctimes%20%28400-1%29%7d%7b2%7d%3d79800) uzunluklu bir öznitelik oluşturduğundan çoğu uygulamada çok büyük bir işlem yükü getirecektir. Bu nedenle IMLAB görüntü işleme kütüphanesinde belirtilen !["N"](https://render.githubusercontent.com/render/math?math=N) öznitelik sayısı kadar nokta çifti verilen girdi imgesinden rastgele örnekleme ile seçilerek NPF öznitelik vektörü oluşturulmuştur.

IMLAB görüntü işleme kütüphanesinde NPF özniteliğinin çıkarılmasını sağlayacak öznitelik sınıfının oluşturulması için aşağıdaki fonksiyon çağrılmalıdır.

```c
struct feature_t *fe = feature_create(CV_NPD, N, M, C, "-n_sample:4000 -min_distance:5 -max_distance:20");
```
Verilen fonksiyonda ilk parametre `CV_NPD` çıkarılacak özniteliği, sonraki üç paramtre `N, M, C` ise sırasıyla görüntü genişliği, görüntü yüksekliği ve kanal sayısını belirtmektedir. Metin olarak verilen argümanlarda ise örnek sayısı !["N=4000"](https://render.githubusercontent.com/render/math?math=N%3d4000) olarak verilmiştir. Orjinal makalede yer almayan `min_distance` ve `max_distance` parametreleri ise piksel farkı hesaplamasında kullanılacak !["(p_i, p_j)"](https://render.githubusercontent.com/render/math?math=%28p_i%2c%20p_j%29) noktaları arasındaki en küçük ve en büyük Öklid uzaklığını belirtmek için kullanılmıştır.

### Yerel İkili Örüntüler (Local Binary Patterns)

Yerel ikili örüntüler; gri seviyeli resimler üzerinde doku ölçümü yapmak için Timo Ojala vd. tarafından 1996 yılında önerilen örüntü tanımlayıcılarıdır. Temelde, bir YİÖ işleminde merkez seçilen bir piksel kerteriz alınarak, komşu piksellerin bu referansa göre durumları sayılara dönüştürülür. Böylelikle yerel bir örüntünün karaktersitik yapısı tek bir piksel içerisinde saklanabilir. Bir YİÖ işleminin matematiksel ifadesi aşağıdaki gibidir. 

!["YIO_{P,R}=\sum_{p=0}^{P-1}u(g_{p,R}-g_c)2^p \label{lbp} \tag{3}"](https://render.githubusercontent.com/render/math?math=YIO_%7bP%2cR%7d%3d%5csum_%7bp%3d0%7d%5e%7bP-1%7du%28g_%7bp%2cR%7d-g_c%292%5ep%20%5clabel%7blbp%7d%20%5ctag%7b3%7d) 

Burada !["u(.)"](https://render.githubusercontent.com/render/math?math=u%28.%29) birim basamak fonksiyon, !["g_c"](https://render.githubusercontent.com/render/math?math=g_c) merkez seçilen ve YİÖ işlecinin uygulandığı piksel, !["g_{p,R}"](https://render.githubusercontent.com/render/math?math=g_%7bp%2cR%7d), !["R"](https://render.githubusercontent.com/render/math?math=R)-ninci komşulukta !["p"](https://render.githubusercontent.com/render/math?math=p)-ninci pikselin değeridir.

Aşağıdaki görselde, !["R=1"](https://render.githubusercontent.com/render/math?math=R%3d1) de bulunan tüm komşuluklar için Denklem \ref{lbp} ile verilen ifadenin nasıl gerçekleştirildiği verilmiştir.

![Yerel İkili Örüntüler#half][lbp]

Verilen görselde (a) bir imgeden seçilen !["3 \times 3"](https://render.githubusercontent.com/render/math?math=3%20%5ctimes%203) boyutunda bir bölgeyi göstermektedir. Bu bölgenin kerteriz noktası imgenin merkez pikselidir. Bu değer (!["g_c=34"](https://render.githubusercontent.com/render/math?math=g_c%3d34)) eşik değeri seçilerek, komşu pikseller bu değere göre eşiklenirse (b) imgesi elde edilir. Denklem \ref{lbp} ile verilen !["2^p"](https://render.githubusercontent.com/render/math?math=2%5ep) ağırlıklandırılması ise (c) imgesinde görselleştirilmiştir. Burada sol üst köşe !["p=0"](https://render.githubusercontent.com/render/math?math=p%3d0) ve !["p"](https://render.githubusercontent.com/render/math?math=p) nin saat yönünde arttığı varsayılmıştır. Son olarak (b) ve (c) imgelerinin çarpılması sonucunda da (d) imgesi elde edilmiştir. Denklem \ref{lbp} ile verilen tanım gereğince de !["g_c"](https://render.githubusercontent.com/render/math?math=g_c) pikseline ait yerel ikili örüntü değeri, (d) imgesinin tüm değrlerinin toplamı, !["YIO_{8,1} = 173"](https://render.githubusercontent.com/render/math?math=YIO_%7b8%2c1%7d%20%3d%20173) elde edilmiştir. 

Verilen bir imgenin !["YIO_{8,1}"](https://render.githubusercontent.com/render/math?math=YIO_%7b8%2c1%7d) özniteliği bulunmak istendiğinde, görüntü içerisinde bulunan tüm !["3 \times 3"](https://render.githubusercontent.com/render/math?math=3%20%5ctimes%203) bölgelere yukarıda detayları verilen işlemlerin uygulanması gerekir. 

Yerel ikili örüntüler özniteliğinin en önemli özelliği görüntünün parlaklık seviyesinden etkilenmemesidir. İşlem her bir öznitelik değerini belirli bir komşulukta yer alan diğer piksellerin değerine göre oluşturduğundan görüntünün parlaklık seviyesinden bağımsızdır. Ancak hesaplamalar piksel tabanlı yapıldığından girdi görüntüsünün bir piksel dahi kayması (sağa,sola,yukarı,aşağı) durumunda aynı piksele karşı gelen öznitelik çok farklı olacaktır. Bunun önüne geçmek için YİÖ öznitelikleri genellikle, literatürde sıklıkla kullanılan, hücre histogramları ile birlikte kullanılır.

Bu yöntemde YİÖ özniteliği hesaplandıktan sonra YİÖ imgesi !["K \times L"](https://render.githubusercontent.com/render/math?math=K%20%5ctimes%20L) boyutunda hücrelere ayrılır ve bu hücrelerin histogramları hesaplanır ve bu histogramlar tüm hücreler için uç uca eklenerek öznitelik vektörü çıkarılır. Böylelikle girdi imgesinde meydana gelen küçük kaymaların yaratacağı olumsuz etkiler azaltılmış olur.

2000 yılında Ojala vd. tarafından ilk makaleye ek olarak önemli bir katkı daha yapılmıştır. Yayınlanan makalede YİÖ özniteliklerinin rotasyon (dönme) işlemine de bağımsız olmasını sağlamak amacıyla hesaplanan değerlerin 'tekdüze' (uinform) veya 'düzensiz' (non-uniform) olarak gruplandırılması önerilmiştir. Bu gruplama hesaplanan (b) imgesindeki sıfır-bir ve bir-sıfır geçişlerinin sayısına bakılarak yapılmıştır. Ojala vd. tarafından geçiş sayısı üçten küçük olan tüm yerel ikili örüntüler tekdüze olarak kabul edilmiştir. Seçilen eşik değeri için 256 farklı ikili örüntüden sadece 36 tanesi 'tekdüze' olarak sınıflandırılmıştır.

Bu gruplandırma sonucunda histogram vektörü; tekdüze değerler ayrık histogram adımlarında, düzensiz tüm örüntülerse tek bir histogram adımına dahil edilerek oluşturulur. 

IMLAB görüntü işleme kütüphanesinde YİÖ özniteliğinin çıkarılmasını sağlayacak öznitelik sınıfının oluşturulması için aşağıdaki fonksiyon çağrılmalıdır.

```c
struct feature_t *fe = feature_create(CV_LBP, N, M, C, "-block:4x4 -uniform:3");
```
Verilen fonksiyonda ilk parametre `CV_LBP` çıkarılacak özniteliği, sonraki üç paramtre `N, M, C` ise sırasıyla görüntü genişliği, görüntü yüksekliği ve kanal sayısını belirtmektedir. Metin olarak verilen argümanlarda ise hücre boyutu !["K \times L = 4\times 4"](https://render.githubusercontent.com/render/math?math=K%20%5ctimes%20L%20%3d%204%5ctimes%204) olarak verilmiştir. Burada yer alan `uniform` değeri hesaplamalarda kullanılacak 'tekdüze' lik kriterini sağlamak için kullanılmaktadır.

### Yönlü Gradyanlar Histogramı (Histogram of Oriented Gradients)

Yönlü gradyanlar histogramı; Navneet Dalal vd. tarafından 2005 yılında önerilen, nesne tanımada sıklıkla kullanılan ve bir nesneyi içerdiği açılar ile betimleyen bir tanımlayıcıdır. Özniteliğin hesaplanabilmesi için öncelikle yönlü gradyan hesaplaması yapılmalıdır. 

!["h=[-1,0,1]"](https://render.githubusercontent.com/render/math?math=h%3d%5b-1%2c0%2c1%5d) olmak üzere verilen bir !["I"](https://render.githubusercontent.com/render/math?math=I) imgesinin !["x"](https://render.githubusercontent.com/render/math?math=x) doğrultusundaki gradyanı !["G_x=I\ast h"](https://render.githubusercontent.com/render/math?math=G_x%3dI%5cast%20h), !["y"](https://render.githubusercontent.com/render/math?math=y) doğrultusundaki gradyanı !["G_y=I\ast (-h)^\intercal"](https://render.githubusercontent.com/render/math?math=G_y%3dI%5cast%20%28-h%29%5e%5cintercal)  denkelmleri ile elde edilir. Bu durumda bir pikselin gradyan şiddeti Denklem \ref{hog_magnitude} ile verilen eşitlik ile ifade edilir.

![" \rho(x,y)=\sqrt{ {G_x(x,y)}^2 + {G_y(x,y)}^2 } \label{hog_magnitude} \tag{4} "](https://render.githubusercontent.com/render/math?math=%20%5crho%28x%2cy%29%3d%5csqrt%7b%20%7bG_x%28x%2cy%29%7d%5e2%20%2b%20%7bG_y%28x%2cy%29%7d%5e2%20%7d%20%5clabel%7bhog_magnitude%7d%20%5ctag%7b4%7d%20)

Bulunan gradyanlar kullanılarak, !["I"](https://render.githubusercontent.com/render/math?math=I) imgesinin !["x,y"](https://render.githubusercontent.com/render/math?math=x%2cy) noktasındaki pikseline ait açı değeri ise Denklem \ref{hog_theta} ile elde edilir.

!["\theta(x,y)=tan^{-1}\left(\frac{G_y(x,y)}{G_x(x,y)}\right) \label{hog_theta} \tag{5}"](https://render.githubusercontent.com/render/math?math=%5ctheta%28x%2cy%29%3dtan%5e%7b-1%7d%5cleft%28%5cfrac%7bG_y%28x%2cy%29%7d%7bG_x%28x%2cy%29%7d%5cright%29%20%5clabel%7bhog_theta%7d%20%5ctag%7b5%7d) 

Elde edilen bu öznitelik her ne kadar önemli bir bilgi olsa da bir pikselin açısı imge gürültüsüne aşırı bağlı olduğundan doğrudan öznitelik olarak kullanılmaya uygun değildir. Navneet Dalal vd. tarafından yapılan en önemli katkı, Denklem \ref{hog_theta} ile verilen özniteliklerin, yerel ikili örüntülere benzer şekilde, !["K \times L"](https://render.githubusercontent.com/render/math?math=K%20%5ctimes%20L) boyutlarında yerel bir hücre içerisinde histogram olarak saklanmasını önermeleridir. Bu öneri ile hem gürültüden kurtulmak, hem de piksel kaymalarının yaratacağı olumsuz etkileri azaltmak mümkün olmuştur.

Denklem \ref{hog_theta} ile verilen açı değerleri !["[0, 360^{\circ}]"](https://render.githubusercontent.com/render/math?math=%5b0%2c%20360%5e%7b%5ccirc%7d%5d) arasında bir sonuç üretebilmektedir. Bu değerlerin herhangi bir kuvantalama yapılmadan histogramının hesaplanabilmesi için sonsuz sayıda histogram adımına ihtiyaç vardır. Ancak Navneet Dalal vd. yaptıkları çalışmada !["B=9"](https://render.githubusercontent.com/render/math?math=B%3d9) adımlı bir histogram hesaplamasının yeterli sonuçları üretebildiğini göstermişlerdir. !["B=9"](https://render.githubusercontent.com/render/math?math=B%3d9) seçilmesi durumunda Denklem \ref{hog_theta} ile hesaplanan açı değerleri, adımları !["[10, 30, 50, \vdots, 150, 170]"](https://render.githubusercontent.com/render/math?math=%5b10%2c%2030%2c%2050%2c%20%5cvdots%2c%20150%2c%20170%5d) olan bir histogram vektöründe saklanmış olacaktır. Referans alınan makalede, histogram adımlarında biriken açı yoğunluğunu hesaplamak içinse, Denklem \ref{hog_magnitude} ile bulunan !["\rho(x,y)"](https://render.githubusercontent.com/render/math?math=%5crho%28x%2cy%29) gradyan şiddetlerinin kullanılması önerilmiştir.

Ancak !["K \times L"](https://render.githubusercontent.com/render/math?math=K%20%5ctimes%20L) boyutlu hücreler üzerinden hesaplanan YGH, gradyan şiddetine yani imge kontrastına bağlı olacaktır. Bunun önüne geçmek için Navneet Dalal vd. her !["K \times L"](https://render.githubusercontent.com/render/math?math=K%20%5ctimes%20L) hücreleri için hesaplanan histogram vektörlerinin !["P \times R"](https://render.githubusercontent.com/render/math?math=P%20%5ctimes%20R) lik bloklar halinde gruplanarak normalize edilmesini önermiştir.

Yukarıda verile işlem adımları kullanılarak YGH hesaplaması şu şekilde yapılmaktadır.

* !["I_{M\times N}"](https://render.githubusercontent.com/render/math?math=I_%7bM%5ctimes%20N%7d) imgesini !["K \times L"](https://render.githubusercontent.com/render/math?math=K%20%5ctimes%20L) boyutundaki hücrelere böl
* Her hücre için !["B_{ij}"](https://render.githubusercontent.com/render/math?math=B_%7bij%7d) uzunluklu !["v_{ij}"](https://render.githubusercontent.com/render/math?math=v_%7bij%7d) yönlü gradyan histogramını hesapla
* !["P \times R"](https://render.githubusercontent.com/render/math?math=P%20%5ctimes%20R) hücre için hesaplanan histogram vektörlerini uç uca ekleyerek !["V_{ij}"](https://render.githubusercontent.com/render/math?math=V_%7bij%7d) blok histogramını bul
* Blok histogramını normalize ederek özniteliği bul !["f_{ij}=\frac{V_{ij}}{\sqrt {\|V_{ij}\|_{2}^{2}+e^{2}}}"](https://render.githubusercontent.com/render/math?math=f_%7bij%7d%3d%5cfrac%7bV_%7bij%7d%7d%7b%5csqrt%20%7b%5c%7cV_%7bij%7d%5c%7c_%7b2%7d%5e%7b2%7d%2be%5e%7b2%7d%7d%7d)

IMLAB görüntü işleme kütüphanesinde YGH özniteliğinin çıkarılmasını sağlayacak öznitelik sınıfının oluşturulması için aşağıdaki fonksiyon çağrılmalıdır.

```c
struct feature_t *fe = feature_create(CV_HOG, N, M, C, "-cell:5x5 -block:2x2 -stride:1x1 -bins:9");
```
Verilen fonksiyonda ilk parametre `CV_HOG` çıkarılacak özniteliği, sonraki üç paramtre `N, M, C` ise sırasıyla görüntü genişliği, görüntü yüksekliği ve kanal sayısını belirtmektedir. Metin olarak verilen argümanlarda ise hücre boyutu !["K \times L = 5\times 5"](https://render.githubusercontent.com/render/math?math=K%20%5ctimes%20L%20%3d%205%5ctimes%205), blok boyutu !["P \times R = 2 \times 2"](https://render.githubusercontent.com/render/math?math=P%20%5ctimes%20R%20%3d%202%20%5ctimes%202) olarak verilmiştir. Burada yer alan `stride` değeri hesaplamalarda kullanılacak bloklamanın hangi sıklıkta yapılacağını göstermektedir. Verilen !["1 \times 1"](https://render.githubusercontent.com/render/math?math=1%20%5ctimes%201) değeri için bloklar yatayda ve dikeyde birer hücre kaydırılarak oluşturulacaktır. `bins` parametresi ise histogram hesaplamasında kullanılacak adım sayısını belirtmek içim kullanılmaktadır.

### Özniteliklerin Kullanılması

Yukarıda detaylı anlatımları yapılan öznitelik çıkarıcılarının örnek bir imgeye uygulanması için aşağıdaki kod parçasının yazılması gerekmektedir.

```c
// read the input image
matrix_t *img = imread("../data/example.bmp");
matrix_t *gray = matrix_create(uint8_t, rows(img), cols(img), 1);

// convert the input into grayscale
rgb2gray(img, gray);

// create feature extractor
struct feature_t *fe = feature_create(CV_HOG, N, M, C, "-cell:5x5 -block:2x2 -stride:1x1 -bins:9");

// allocate space for the feature vector
float *feature = (float *)calloc(feature_size(fe), sizeof(float));

// display information about the feature (optional)
feature_view(fe);

// print the class names and files under them
feature_extract(gray, fe, feature);
```

Burada bellekten okunan renkli bir imge öncelikle gri seviyeye dönüştürülmüştür. Ardından çıkarılma istenen öznitelik sınıfından (`CV_HOG`) bir eleman oluşturulmuştur. Çıkarılmak istenen öznitelik için gerekli boyut bilgisi `feature_size` fonksiyonu yardımı ile öğrenilerek gerekli alan ayrınmıştır. Ardından `feature_extract` fonksiyonu ile verilen gri imgeden `fe` öznitelik çıkarıcı sınıfı kullanılarak öznitelikler çıkarılmış ve `feature` alanına yazılmıştır.

Yazıda yer alan analizlerin yapıldığı kod parçaları, görseller ve kullanılan veri setlerine [imlab_feature_extraction](https://github.com/cescript/imlab_feature_extraction) GitHub sayfası üzerinden erişilebilir.

**Referanslar**
* Liao, Shengcai, Anil K. Jain, and Stan Z. Li. "A fast and accurate unconstrained face detector." IEEE transactions on pattern analysis and machine intelligence 38.2 (2015): 211-223.

* Ojala, Timo, Matti Pietikäinen, and David Harwood. "A comparative study of texture measures with classification based on featured distributions." Pattern recognition 29.1 (1996): 51-59.

* Ojala, Timo, Matti Pietikäinen, and Topi Mäenpää. "Gray scale and rotation invariant texture classification with local binary patterns." European Conference on Computer Vision. Springer, Berlin, Heidelberg, 2000.

* Dalal, Navneet, and Bill Triggs. "Histograms of oriented gradients for human detection." 2005 IEEE computer society conference on computer vision and pattern recognition (CVPR'05). Vol. 1. IEEE, 2005.


[RESOURCES]: # (List of the resources used by the blog post)
[example]: /assets/post_resources/feature_extraction/example.png
[example_dark]: /assets/post_resources/feature_extraction/example_dark.png
[encoder_result]: /assets/post_resources/feature_extraction/encoder_result.png
[encoder_result_dark]: /assets/post_resources/feature_extraction/encoder_result_dark.png
[npd_result]: /assets/post_resources/feature_extraction/npd_result.png
[npd_result_dark]: /assets/post_resources/feature_extraction/npd_result_dark.png
[lbp_result]: /assets/post_resources/feature_extraction/lbp_result.png
[lbp_result_dark]: /assets/post_resources/feature_extraction/lbp_result_dark.png
[hog_result]: /assets/post_resources/feature_extraction/hog_result.png
[hog_result_dark]: /assets/post_resources/feature_extraction/hog_result_dark.png
[lbp]: /assets/post_resources/feature_extraction/lbp.png
