---
layout: post
title: Öznitelik Çıkarma
slug: feature-extraction
author: Bahri ABACI
categories:
- Görüntü İşleme
- Makine Öğrenmesi
references: ""
thumbnail: /assets/post_resources/feature_extraction/thumbnail.png
---

Bilgisayarlı görü uygulamaları yaygın olarak iki temel aşamadan oluşmaktadır. Bunlardan ilki olan Öznitelik Çıkarımı (Feature Extraction) aşamasında verilen girdi imgesinden; parlaklık, zıtlık, dönme gibi kamera ve ışık koşullarından bağımsız olarak girdi imgesini betimleyen tanımlayıcıların (özniteliklerin) çıkarılması hedeflenmektedir. İkinci aşama olan Makine Öğrenmesi (Machine Learning) aşamasında ise girdi imgesinden elde edilen öznitelikler kullanılarak verinin sınıflandırılması hedeflenmektedir. Blogda paylaştığımız [K-Means]({% post_url 2015-08-28-k-means-kumeleme-algoritmasi %}), [Temel Bileşen Analizi]({% post_url 2019-09-01-temel-bilesen-analizi %}), [Lojistik Regresyon Analizi]({% post_url 2015-07-23-lojistik-regresyon-analizi %}) ve [Karar Ağaçları]({% post_url 2015-11-01-karar-agaclari %}) gibi eski yazılarımızda Makine Öğrenmesi aşaması ile ilgili pek çok yöntemi incelemiştik. Bu yazımızda ise IMLAB görüntü işleme kütüphanesi içerisinde yer alan ve Öznitelik Çıkarımı için kullanılan yöntemleri inceleyeceğiz.

<!--more-->

Yazıda incelenecek yöntemlere geçmeden önce öznitelik kavramına ve bilgisayarlı görüdeki önemini anlamaya çalışalım. Aşağıda ilk sütunda verilen iki imge insan gözü için aynı aynı sahneyi gösterse de bilgisayarlar açısından görüntü pikseller ve bunun değerlerinden ibaret olduğundan birbirinden tamamen farklı iki imge olarak algılanacaktır. Öznitelik çıkarma işlemi; piksel değerlerini doğrudan kullanmak yerine, insan gözününde yaptığı gibi, pikseller arasındaki komşuluk ilişkisini kodlamayı amaçlamaktadır. Böylece piksel değerleri ne olursa olsun, pikseller arasındaki ilişki korunduğu müddetçe iki imge için de üretilen kod sözcükleri aynı olacak ve makine öğrenmesi için uygun bir veri hazırlanacaktır.

| Örnek İmge |  Gri Seviye Kodlama | Normalize Piksel Farkları | Yerel İkili Örüntüler | Yönlü Gradyanlar Histogramı |
:-------:|:----:|:----:|:---:|:---:|
![Örnek İmge][example] | ![Örnek İmge][encoder_result] | ![Normalize Piksel Farkları][npd_result] | ![Yerel İkili Örüntüler][lbp_result] | ![Yönlü Gradyanlar Histogramı][hog_result]
![Örnek İmge][example_dark] | ![Örnek İmge][encoder_result_dark] | ![Normalize Piksel Farkları][npd_result_dark] | ![Yerel İkili Örüntüler][lbp_result_dark] | ![Yönlü Gradyanlar Histogramı][hog_result_dark]

Yukarıda verilen tabloda (2. sütundan itibaren) IMLAB görüntü işleme kütüphanesi kullanılarak çıkarılabilecek öznitelikler görselleştirilmiştir. Tablodan da görüldüğü üzere, girdi görüntüsünde yüksek bir ışık değişimi dahi meydana gelse, iki imge için de çıkarılan öznitelikler büyük derecede ortaklık göstermektedir. 

Aşağıda tabloda verilen ve IMLAB görüntü işleme kütüphanesinde yer alan öznitelik çıkarıcı yöntemler basitten karmaşığa doğru sıralanarak açıklanmıştır.

### Gri Seviye Kodlama (Grayscale)

Gri seviye kodlama imge işlemede kullanılan en basit özniteliktir. Bu kodlama öznitelik çıkarımının amacına her ne kadar aykırı olsa da pikseller arasındaki ilişkiyi öğrenme potansiyeli olan karmaşık makine öğrenme algoritmalarının (Yapay Sinir Ağları, Evrişimsel Sinir Ağları gibi) kullanılması durumunda hala tercih edilebilmektedir. Kırmızı (R), Yeşil (G) ve Mavi (B) gibi üç kanaldan oluşan <img src="assets/post_resources/math//becb74c6f69bb68933ac4c568a4fe9e2.svg?invert_in_darkmode" align=middle width=59.74221494999998pt height=22.465723500000017pt/> imgesi için gri seviye öznitelikler aşağıdaki formül ile hesaplanır.

<p align="center"><img src="assets/post_resources/math//72a0a5959cc9545997af90684a085ea2.svg?invert_in_darkmode" align=middle width=364.67639999999994pt height=34.7253258pt/></p>

IMLAB görüntü işleme kütüphanesinde gri seviye özniteliğinin çıkarılmasını sağlayacak öznitelik sınıfının oluşturulması için aşağıdaki fonksiyon çağrılmalıdır.

```c
struct feature_t *fe = feature_create(CV_ENCODER, N, M, C, "");
```
Verilen fonksiyonda ilk parametre `CV_ENCODER` çıkarılacak özniteliği, sonraki üç paramtre `N, M, C` ise sırasıyla görüntü genişliği, görüntü yüksekliği ve kanal sayısını belirtmektedir. Metin olarak verilen argüman ise öznitelik çıkarma algoritmasına özel parametrelerin fonksiyona geçirilmesi için eklenmiştir. `CV_ENCODER` kodlayıcısı herhangi bir paarametreye ihtiyaç duymadığından boş olarak bırakılmıştır.

### Normalize Piksel Farkları (Normalized Pixel Difference)

Normalize piksel farkları; gri seviye görüntülerde öznitelik çıkarımı için Shengcai Liao vd. tarafından 2015 yılında önerilen özniteliklerdir. NPF özniteliğinde iki piksel arasındaki göreceli fark ölçülmektedir. <img src="assets/post_resources/math//cadea9c8271a77de38e1deb11b924e68.svg?invert_in_darkmode" align=middle width=42.91566839999999pt height=22.465723500000017pt/> imgesi içerisinde seçilen bir adet <img src="assets/post_resources/math//1b6ae11533e15e133cdd358dc7b133da.svg?invert_in_darkmode" align=middle width=179.97134924999997pt height=24.65753399999998pt/> nokta çifti için <img src="assets/post_resources/math//69f558d8c6f5865cde5fab4cb26f1a04.svg?invert_in_darkmode" align=middle width=66.39130739999999pt height=22.465723500000017pt/> özniteliği aşağıdaki formül ile hesaplanır.

<p align="center"><img src="assets/post_resources/math//b8fc0f1d81e1aa2b403a4acdf6913fbc.svg?invert_in_darkmode" align=middle width=228.93825405pt height=39.428498999999995pt/></p> 

Burada denklemin pay kısmında yazılı olan terim ile iki piksel arasındaki büyüklük küçüklük ilişkisi kodlanırken, bölme işlemi ile de işlemin görüntünün parlaklığa olan duyarlılığı azaltılmaktadır. Payda teriminin sıfır olması durumunda ise iki piksel arasında herhangi bir fark bulunmadığından <img src="assets/post_resources/math//5be41b7fea2159f47839316ac124706b.svg?invert_in_darkmode" align=middle width=86.37549029999998pt height=22.465723500000017pt/> kullanılması önerilmiştir.

Shengcai Liao vd. tarafından önerilen makalede, NPF özniteliklerinin, <img src="assets/post_resources/math//cadea9c8271a77de38e1deb11b924e68.svg?invert_in_darkmode" align=middle width=42.91566839999999pt height=22.465723500000017pt/> imgesi içerisinde seçilebilecek tüm ikili piksel grupları kullanılarak hesaplanması önerilmiştir. Ancak bu <img src="assets/post_resources/math//48899edacbdd8785ebce00e73f66dfbf.svg?invert_in_darkmode" align=middle width=52.968029399999985pt height=21.18721440000001pt/> bir imge için dahi <img src="assets/post_resources/math//48fa3178cbb133bcb300a62ffb4f651e.svg?invert_in_darkmode" align=middle width=180.56628479999998pt height=33.20539859999999pt/> uzunluklu bir öznitelik oluşturduğundan çoğu uygulamada çok büyük bir işlem yükü getirecektir. Bu nedenle IMLAB görüntü işleme kütüphanesinde belirtilen <img src="assets/post_resources/math//f9c4988898e7f532b9f826a75014ed3c.svg?invert_in_darkmode" align=middle width=14.99998994999999pt height=22.465723500000017pt/> öznitelik sayısı kadar nokta çifti verilen girdi imgesinden rastgele örnekleme ile seçilerek NPF öznitelik vektörü oluşturulmuştur.

IMLAB görüntü işleme kütüphanesinde NPF özniteliğinin çıkarılmasını sağlayacak öznitelik sınıfının oluşturulması için aşağıdaki fonksiyon çağrılmalıdır.

```c
struct feature_t *fe = feature_create(CV_NPD, N, M, C, "-n_sample:4000 -min_distance:5 -max_distance:20");
```
Verilen fonksiyonda ilk parametre `CV_NPD` çıkarılacak özniteliği, sonraki üç paramtre `N, M, C` ise sırasıyla görüntü genişliği, görüntü yüksekliği ve kanal sayısını belirtmektedir. Metin olarak verilen argümanlarda ise örnek sayısı <img src="assets/post_resources/math//944a8c5aa68fb46486b90dc81f2a2d56.svg?invert_in_darkmode" align=middle width=69.79443734999998pt height=22.465723500000017pt/> olarak verilmiştir. Orjinal makalede yer almayan `min_distance` ve `max_distance` parametreleri ise piksel farkı hesaplamasında kullanılacak <img src="assets/post_resources/math//9d76d028496beffb0440c7a229db7800.svg?invert_in_darkmode" align=middle width=49.03165079999999pt height=24.65753399999998pt/> noktaları arasındaki en küçük ve en büyük Öklid uzaklığını belirtmek için kullanılmıştır.

### Yerel İkili Örüntüler (Local Binary Patterns)

Yerel ikili örüntüler; gri seviyeli resimler üzerinde doku ölçümü yapmak için Timo Ojala vd. tarafından 1996 yılında önerilen örüntü tanımlayıcılarıdır. Temelde, bir YİÖ işleminde merkez seçilen bir piksel kerteriz alınarak, komşu piksellerin bu referansa göre durumları sayılara dönüştürülür. Böylelikle yerel bir örüntünün karaktersitik yapısı tek bir piksel içerisinde saklanabilir. Bir YİÖ işleminin matematiksel ifadesi aşağıdaki gibidir. 

<p align="center"><img src="assets/post_resources/math//992d0e88e43a1cf17b867bbc6db05774.svg?invert_in_darkmode" align=middle width=210.40924574999997pt height=49.84491765pt/></p> 

Burada <img src="assets/post_resources/math//722749020f01ee151c681c7f818fb275.svg?invert_in_darkmode" align=middle width=26.761930799999988pt height=24.65753399999998pt/> birim basamak fonksiyon, <img src="assets/post_resources/math//eb30b7762aec8ad6b9c53cf51888ac16.svg?invert_in_darkmode" align=middle width=13.71523394999999pt height=14.15524440000002pt/> merkez seçilen ve YİÖ işlecinin uygulandığı piksel, <img src="assets/post_resources/math//cdd10975b67ee1becd4a8f7200532e96.svg?invert_in_darkmode" align=middle width=28.483011149999992pt height=14.15524440000002pt/>, <img src="assets/post_resources/math//1e438235ef9ec72fc51ac5025516017c.svg?invert_in_darkmode" align=middle width=12.60847334999999pt height=22.465723500000017pt/>-ninci komşulukta <img src="assets/post_resources/math//2ec6e630f199f589a2402fdf3e0289d5.svg?invert_in_darkmode" align=middle width=8.270567249999992pt height=14.15524440000002pt/>-ninci pikselin değeridir.

Aşağıdaki görselde, <img src="assets/post_resources/math//e4d2d8a7b8fb2790d2713048ef37e918.svg?invert_in_darkmode" align=middle width=42.74531084999999pt height=22.465723500000017pt/> de bulunan tüm komşuluklar için Denklem \ref{lbp} ile verilen ifadenin nasıl gerçekleştirildiği verilmiştir.

![Yerel İkili Örüntüler#half][lbp]

Verilen görselde (a) bir imgeden seçilen <img src="assets/post_resources/math//9f2b6b0a7f3d99fd3f396a1515926eb3.svg?invert_in_darkmode" align=middle width=36.52961069999999pt height=21.18721440000001pt/> boyutunda bir bölgeyi göstermektedir. Bu bölgenin kerteriz noktası imgenin merkez pikselidir. Bu değer (<img src="assets/post_resources/math//3e60163227a8b17a446566ed27019b74.svg?invert_in_darkmode" align=middle width=52.893200249999985pt height=21.18721440000001pt/>) eşik değeri seçilerek, komşu pikseller bu değere göre eşiklenirse (b) imgesi elde edilir. Denklem \ref{lbp} ile verilen <img src="assets/post_resources/math//d0eaea7c205f9c79730b9aa5337c89fd.svg?invert_in_darkmode" align=middle width=14.995686749999992pt height=21.839370299999988pt/> ağırlıklandırılması ise (c) imgesinde görselleştirilmiştir. Burada sol üst köşe <img src="assets/post_resources/math//a32f79dc69538108b6040ac855993285.svg?invert_in_darkmode" align=middle width=38.40740639999999pt height=21.18721440000001pt/> ve <img src="assets/post_resources/math//2ec6e630f199f589a2402fdf3e0289d5.svg?invert_in_darkmode" align=middle width=8.270567249999992pt height=14.15524440000002pt/> nin saat yönünde arttığı varsayılmıştır. Son olarak (b) ve (c) imgelerinin çarpılması sonucunda da (d) imgesi elde edilmiştir. Denklem \ref{lbp} ile verilen tanım gereğince de <img src="assets/post_resources/math//eb30b7762aec8ad6b9c53cf51888ac16.svg?invert_in_darkmode" align=middle width=13.71523394999999pt height=14.15524440000002pt/> pikseline ait yerel ikili örüntü değeri, (d) imgesinin tüm değrlerinin toplamı, <img src="assets/post_resources/math//22c76f8cea349c3240d3edcbdf64347c.svg?invert_in_darkmode" align=middle width=98.6575062pt height=22.465723500000017pt/> elde edilmiştir. 

Verilen bir imgenin <img src="assets/post_resources/math//e44e16d04324c69b04c30cfd390111ee.svg?invert_in_darkmode" align=middle width=51.26036189999999pt height=22.465723500000017pt/> özniteliği bulunmak istendiğinde, görüntü içerisinde bulunan tüm <img src="assets/post_resources/math//9f2b6b0a7f3d99fd3f396a1515926eb3.svg?invert_in_darkmode" align=middle width=36.52961069999999pt height=21.18721440000001pt/> bölgelere yukarıda detayları verilen işlemlerin uygulanması gerekir. 

Yerel ikili örüntüler özniteliğinin en önemli özelliği görüntünün parlaklık seviyesinden etkilenmemesidir. İşlem her bir öznitelik değerini belirli bir komşulukta yer alan diğer piksellerin değerine göre oluşturduğundan görüntünün parlaklık seviyesinden bağımsızdır. Ancak hesaplamalar piksel tabanlı yapıldığından girdi görüntüsünün bir piksel dahi kayması (sağa,sola,yukarı,aşağı) durumunda aynı piksele karşı gelen öznitelik çok farklı olacaktır. Bunun önüne geçmek için YİÖ öznitelikleri genellikle, literatürde sıklıkla kullanılan, hücre histogramları ile birlikte kullanılır.

Bu yöntemde YİÖ özniteliği hesaplandıktan sonra YİÖ imgesi <img src="assets/post_resources/math//ba4d5d21c97d8488fa733e12b201b3ed.svg?invert_in_darkmode" align=middle width=46.41543389999999pt height=22.465723500000017pt/> boyutunda hücrelere ayrılır ve bu hücrelerin histogramları hesaplanır ve bu histogramlar tüm hücreler için uç uca eklenerek öznitelik vektörü çıkarılır. Böylelikle girdi imgesinde meydana gelen küçük kaymaların yaratacağı olumsuz etkiler azaltılmış olur.

2000 yılında Ojala vd. tarafından ilk makaleye ek olarak önemli bir katkı daha yapılmıştır. Yayınlanan makalede YİÖ özniteliklerinin rotasyon (dönme) işlemine de bağımsız olmasını sağlamak amacıyla hesaplanan değerlerin 'tekdüze' (uinform) veya 'düzensiz' (non-uniform) olarak gruplandırılması önerilmiştir. Bu gruplama hesaplanan (b) imgesindeki sıfır-bir ve bir-sıfır geçişlerinin sayısına bakılarak yapılmıştır. Ojala vd. tarafından geçiş sayısı üçten küçük olan tüm yerel ikili örüntüler tekdüze olarak kabul edilmiştir. Seçilen eşik değeri için 256 farklı ikili örüntüden sadece 36 tanesi 'tekdüze' olarak sınıflandırılmıştır.

Bu gruplandırma sonucunda histogram vektörü; tekdüze değerler ayrık histogram adımlarında, düzensiz tüm örüntülerse tek bir histogram adımına dahil edilerek oluşturulur. 

IMLAB görüntü işleme kütüphanesinde YİÖ özniteliğinin çıkarılmasını sağlayacak öznitelik sınıfının oluşturulması için aşağıdaki fonksiyon çağrılmalıdır.

```c
struct feature_t *fe = feature_create(CV_LBP, N, M, C, "-block:4x4 -uniform:3");
```
Verilen fonksiyonda ilk parametre `CV_LBP` çıkarılacak özniteliği, sonraki üç paramtre `N, M, C` ise sırasıyla görüntü genişliği, görüntü yüksekliği ve kanal sayısını belirtmektedir. Metin olarak verilen argümanlarda ise hücre boyutu <img src="assets/post_resources/math//ef8bbd6243e14d95158ae0ec79e47c70.svg?invert_in_darkmode" align=middle width=104.86267439999999pt height=22.465723500000017pt/> olarak verilmiştir. Burada yer alan `uniform` değeri hesaplamalarda kullanılacak 'tekdüze' lik kriterini sağlamak için kullanılmaktadır.

### Yönlü Gradyanlar Histogramı (Histogram of Oriented Gradients)

Yönlü gradyanlar histogramı; Navneet Dalal vd. tarafından 2005 yılında önerilen, nesne tanımada sıklıkla kullanılan ve bir nesneyi içerdiği açılar ile betimleyen bir tanımlayıcıdır. Özniteliğin hesaplanabilmesi için öncelikle yönlü gradyan hesaplaması yapılmalıdır. 

<img src="assets/post_resources/math//c392b3ae2b980f0a464cb7b25bc006cd.svg?invert_in_darkmode" align=middle width=92.576022pt height=24.65753399999998pt/> olmak üzere verilen bir <img src="assets/post_resources/math//21fd4e8eecd6bdf1a4d3d6bd1fb8d733.svg?invert_in_darkmode" align=middle width=8.515988249999989pt height=22.465723500000017pt/> imgesinin <img src="assets/post_resources/math//332cc365a4987aacce0ead01b8bdcc0b.svg?invert_in_darkmode" align=middle width=9.39498779999999pt height=14.15524440000002pt/> doğrultusundaki gradyanı <img src="assets/post_resources/math//244dd6753d3c16868688c04dc5787add.svg?invert_in_darkmode" align=middle width=76.63058864999999pt height=22.831056599999986pt/>, <img src="assets/post_resources/math//deceeaf6940a8c7a5a02373728002b0f.svg?invert_in_darkmode" align=middle width=8.649225749999989pt height=14.15524440000002pt/> doğrultusundaki gradyanı <img src="assets/post_resources/math//6b412c88cd380b3c3872a0447a7faf6f.svg?invert_in_darkmode" align=middle width=109.26965114999999pt height=24.65753399999998pt/>  denkelmleri ile elde edilir. Bu durumda bir pikselin gradyan şiddeti Denklem \ref{hog_magnitude} ile verilen eşitlik ile ifade edilir.

<p align="center"><img src="assets/post_resources/math//e3cea8b27279e32cede2d0c9ade79abc.svg?invert_in_darkmode" align=middle width=238.1285082pt height=29.58934275pt/></p>

Bulunan gradyanlar kullanılarak, <img src="assets/post_resources/math//21fd4e8eecd6bdf1a4d3d6bd1fb8d733.svg?invert_in_darkmode" align=middle width=8.515988249999989pt height=22.465723500000017pt/> imgesinin <img src="assets/post_resources/math//0acac2a2d5d05a8394e21a70a71041b4.svg?invert_in_darkmode" align=middle width=25.350096749999988pt height=14.15524440000002pt/> noktasındaki pikseline ait açı değeri ise Denklem \ref{hog_theta} ile elde edilir.

<p align="center"><img src="assets/post_resources/math//87cc65cce7d9fe5a917587b0b7bf1b7a.svg?invert_in_darkmode" align=middle width=200.5895133pt height=39.452455349999994pt/></p> 

Elde edilen bu öznitelik her ne kadar önemli bir bilgi olsa da bir pikselin açısı imge gürültüsüne aşırı bağlı olduğundan doğrudan öznitelik olarak kullanılmaya uygun değildir. Navneet Dalal vd. tarafından yapılan en önemli katkı, Denklem \ref{hog_theta} ile verilen özniteliklerin, yerel ikili örüntülere benzer şekilde, <img src="assets/post_resources/math//ba4d5d21c97d8488fa733e12b201b3ed.svg?invert_in_darkmode" align=middle width=46.41543389999999pt height=22.465723500000017pt/> boyutlarında yerel bir hücre içerisinde histogram olarak saklanmasını önermeleridir. Bu öneri ile hem gürültüden kurtulmak, hem de piksel kaymalarının yaratacağı olumsuz etkileri azaltmak mümkün olmuştur.

Denklem \ref{hog_theta} ile verilen açı değerleri <img src="assets/post_resources/math//9d8241666276f26f7466df9fdb3c9a25.svg?invert_in_darkmode" align=middle width=56.87225774999999pt height=24.65753399999998pt/> arasında bir sonuç üretebilmektedir. Bu değerlerin herhangi bir kuvantalama yapılmadan histogramının hesaplanabilmesi için sonsuz sayıda histogram adımına ihtiyaç vardır. Ancak Navneet Dalal vd. yaptıkları çalışmada <img src="assets/post_resources/math//55bf7a8e83a9170b69e35072976e94e2.svg?invert_in_darkmode" align=middle width=43.43024399999999pt height=22.465723500000017pt/> adımlı bir histogram hesaplamasının yeterli sonuçları üretebildiğini göstermişlerdir. <img src="assets/post_resources/math//55bf7a8e83a9170b69e35072976e94e2.svg?invert_in_darkmode" align=middle width=43.43024399999999pt height=22.465723500000017pt/> seçilmesi durumunda Denklem \ref{hog_theta} ile hesaplanan açı değerleri, adımları <img src="assets/post_resources/math//d73c31aba25bd5a59a2b58fdc54df5f0.svg?invert_in_darkmode" align=middle width=148.8586011pt height=29.771689199999994pt/> olan bir histogram vektöründe saklanmış olacaktır. Referans alınan makalede, histogram adımlarında biriken açı yoğunluğunu hesaplamak içinse, Denklem \ref{hog_magnitude} ile bulunan <img src="assets/post_resources/math//8a1f734d90dca3edfd2ea1fe9b2aee5f.svg?invert_in_darkmode" align=middle width=46.63439549999998pt height=24.65753399999998pt/> gradyan şiddetlerinin kullanılması önerilmiştir.

Ancak <img src="assets/post_resources/math//ba4d5d21c97d8488fa733e12b201b3ed.svg?invert_in_darkmode" align=middle width=46.41543389999999pt height=22.465723500000017pt/> boyutlu hücreler üzerinden hesaplanan YGH, gradyan şiddetine yani imge kontrastına bağlı olacaktır. Bunun önüne geçmek için Navneet Dalal vd. her <img src="assets/post_resources/math//ba4d5d21c97d8488fa733e12b201b3ed.svg?invert_in_darkmode" align=middle width=46.41543389999999pt height=22.465723500000017pt/> hücreleri için hesaplanan histogram vektörlerinin <img src="assets/post_resources/math//2ba4f7fdc5c3711c62fa93ce36de2394.svg?invert_in_darkmode" align=middle width=45.536435999999995pt height=22.465723500000017pt/> lik bloklar halinde gruplanarak normalize edilmesini önermiştir.

Yukarıda verile işlem adımları kullanılarak YGH hesaplaması şu şekilde yapılmaktadır.

* <img src="assets/post_resources/math//cadea9c8271a77de38e1deb11b924e68.svg?invert_in_darkmode" align=middle width=42.91566839999999pt height=22.465723500000017pt/> imgesini <img src="assets/post_resources/math//ba4d5d21c97d8488fa733e12b201b3ed.svg?invert_in_darkmode" align=middle width=46.41543389999999pt height=22.465723500000017pt/> boyutundaki hücrelere böl
* Her hücre için <img src="assets/post_resources/math//e4e4d0eb927a8c788c0d5f1b819546ad.svg?invert_in_darkmode" align=middle width=23.22406019999999pt height=22.465723500000017pt/> uzunluklu <img src="assets/post_resources/math//1af6ca771de4768bed85ef9cf993e529.svg?invert_in_darkmode" align=middle width=18.72347564999999pt height=14.15524440000002pt/> yönlü gradyan histogramını hesapla
* <img src="assets/post_resources/math//2ba4f7fdc5c3711c62fa93ce36de2394.svg?invert_in_darkmode" align=middle width=45.536435999999995pt height=22.465723500000017pt/> hücre için hesaplanan histogram vektörlerini uç uca ekleyerek <img src="assets/post_resources/math//357759b17b5b6fd4f41e1397054a9d62.svg?invert_in_darkmode" align=middle width=20.344460399999992pt height=22.465723500000017pt/> blok histogramını bul
* Blok histogramını normalize ederek özniteliği bul <img src="assets/post_resources/math//762a5cf9364328c01a10e85cb32617d7.svg?invert_in_darkmode" align=middle width=118.13323334999997pt height=32.40174300000001pt/>

IMLAB görüntü işleme kütüphanesinde YGH özniteliğinin çıkarılmasını sağlayacak öznitelik sınıfının oluşturulması için aşağıdaki fonksiyon çağrılmalıdır.

```c
struct feature_t *fe = feature_create(CV_HOG, N, M, C, "-cell:5x5 -block:2x2 -stride:1x1 -bins:9");
```
Verilen fonksiyonda ilk parametre `CV_HOG` çıkarılacak özniteliği, sonraki üç paramtre `N, M, C` ise sırasıyla görüntü genişliği, görüntü yüksekliği ve kanal sayısını belirtmektedir. Metin olarak verilen argümanlarda ise hücre boyutu <img src="assets/post_resources/math//751dc15bf42f2c2d01f7176dd44ecc2a.svg?invert_in_darkmode" align=middle width=104.86267439999999pt height=22.465723500000017pt/>, blok boyutu <img src="assets/post_resources/math//5ea4ddba81c325cfa0785634c39f73ef.svg?invert_in_darkmode" align=middle width=103.98367319999997pt height=22.465723500000017pt/> olarak verilmiştir. Burada yer alan `stride` değeri hesaplamalarda kullanılacak bloklamanın hangi sıklıkta yapılacağını göstermektedir. Verilen <img src="assets/post_resources/math//d12ef2fd91d2a991710509cae7229134.svg?invert_in_darkmode" align=middle width=36.52961069999999pt height=21.18721440000001pt/> değeri için bloklar yatayda ve dikeyde birer hücre kaydırılarak oluşturulacaktır. `bins` parametresi ise histogram hesaplamasında kullanılacak adım sayısını belirtmek içim kullanılmaktadır.

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
