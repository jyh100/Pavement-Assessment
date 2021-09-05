# Highway Pavement Assessment via Google Earth Web
https://www.yuhanjiang.com/research/IM/PA

SEE THE Image Capture DEMO VIA https://www.yuhanjiang.com/research/IM/PA/GE

SEE THE GPS-ORC DEMO VIA https://www.yuhanjiang.com/research/IM/PA/GE/ORC

Dataset: https://drive.google.com/file/d/1gLwvOAOWaE8Bhk9wdAItSZ87Jvooas-f/view?usp=sharing

@article{doi:10.1061/JPEODX.0000282,
author = {Yuhan Jiang  and Sisi Han  and Yong Bai },
title = {Development of a Pavement Evaluation Tool Using Aerial Imagery and Deep Learning},
journal = {Journal of Transportation Engineering, Part B: Pavements},
volume = {147},
number = {3},
pages = {04021027},
year = {2021},
doi = {10.1061/JPEODX.0000282}
abstract = { This paper presents the research results of using Google Earth imagery for visual condition surveying of highway pavement in the United States. A screenshot tool is developed to automatically track the highway for collecting end-to-end images and Global Position System (GPS). A highway segmentation tool based on a deep convolutional neural network (DCNN) is developed to segment the collected highway images into the predefined object categories, where the cracks are identified and labeled in each small patch of the overlapping assembled label-image prediction. Then, the longitudinal cracks and transverse cracks are detected using the x-gradient and y-gradient from the Sobel operator, and the developed pavement evaluation tool rates the longitudinal cracking in 0.3048  m/30.48  m-Station (linear feet per 100\&nbsp;ft. station) and transverse cracking in number per 30.48  m-Station (100\&nbsp;ft. station), which can be visualized in ArcGIS Online. Experiments were conducted on Interstate 43 (I-43) in Milwaukee County with pavement in both defective and sound visual conditions. Experimental results showed the patch-wise highway segmentation in Google Earth imagery from the 16×16-pixel DCNN model has as precise pixel accuracy as the U-net-based pixelwise crack/noncrack classifier. Compared to the manually crafted label image in the experimental area, the rated longitudinal cracking has an average error of overrating 20\%, while transverse cracking has an average error of underrating 7\%. This research project contributes to visual pavement condition surveying methodology with the free-to-access Google Earth imagery, which is a feasible, cost-effective option for accurately rating and geographically visualizing both project-level and network-level pavement. }
