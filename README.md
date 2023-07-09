
# Multi-view Datasets
## 1. Usage
All dataset files contain two attributes, `X` and `y`. 
- `X` is the multi-view data as a cell, each element in this cell is an _N_-by-_D_ matrix _X<sub>k</sub>_, where _N_ is the number of data points and _D_ is the feature dimensions. So rows of _X<sub>k</sub>_ correspond to data points.
- `y` is the vector of ground-truth labels.

**Note: Due to Github's limitation on file size, files larger than 20MB will be stored in other repositories**：
1. [Baidu](https://pan.baidu.com/s/1vAIi4Tdhuk9c3XNf-48_4Q),  code：yeog
2. [Google](https://drive.google.com/drive/folders/1wkQA0z-cAKVzFIk1izOALCXdqdwsPvx2?usp=share_link)

## 2. Dataset Details
| Dataset | Instances | Clusters | Views(dimension) | Description | Type           |
|---------|-----------|----------|------------------|-------------|----------------|
| 3Sources | 169 | 6 | Reuters(3068), BBC(3560), Guardian(3631) | A new multi-view text dataset collected from three well-known online news sources: [BBC](http://mlg.ucd.ie/datasets/news.bbc.co.uk), [Reuters](http://reuters.co.uk/),and [The Guardian](http://www.guardian.co.uk/) [[14]](#weiFuzzyClusteringMultiview2022). | Text           |
| 100Leaves | 1600 | 100 | SD(64), FSM(64), TH(64) | Sixteen samples of leaf each of one-hundred plant species [[16]](#yangMultiViewAdjacencyConstrainedHierarchical2022). | Image          |
| ACM | 3025 | 3 | View1(1870), View2(3025), View2(3025), View2(3025), View2(3025) | The dataset extracts papers published in KDD, SIGMOD, SIGCOMM, MobiCOMM, and VLDB. |                |
| ALOI-1k | 110250 | 1000 | CS(77), Haralick(13), HSV(64), RGB(125) | The [Amsterdam Library of Object Images](http://aloi.science.uva.nl/) is a collection of 110250 images of 1000 small objects, taken under various light conditions and rotation angles [[25]](#zhangFacilitatedLowrankMultiview2023). | Object         |
| ALOI | 10800 | 100 | CS(77), Haralick(13), HSV(64), RGB(125) | A subset of ALOI-1k [[25]](#zhangFacilitatedLowrankMultiview2023). | Object         |
| Animal | 11673 | 20 | View1(2688), View2(2000), View3(2001), View4(2000), | A selected set of the Animals with Attributes dataset [[12]](#lampertLearningDetectUnseen2009). | Animal         |
| BBCSport | 544 | 5 | View1(3183), View2(3203) | This document dataset contains 544 documents  from the BBC Sport website, and they are aboutthe sports news between 2004 and 2005 [[4]](#chenLowrankTensorBased2022). | Text           |
| BBC4view | 685 | 5 | View1(4659), View2(4633), View3(4665), View4(4684) | Similar to BBCSport, this dataset consists of 685 documents from the BBC Sport website about sports news [[4]](#chenLowrankTensorBased2022). | Text           |
| COIL20 | 1440 | 20 | Intensity(1024), LBP(944), Gabor(4096) | It is the abbreviation of the Columbia object image library dataset [[24]](#wuEssentialTensorLearning2019). | Object         |
| Caltech101-7 | 1474 | 7 | GABOR(48), WM(40), CENT(254), HOG(1984), GIST(512), LBP(928) | This dataset contains 1474 images belonging to seven classes, which are faces, motorbikes, dollar bill, Garfield, stop sign, and windsor chair [[4]](#chenLowrankTensorBased2022). | Object         |
| Caltech101-20 | 2386 | 20 | GABOR(48), WM(40), CENT(254), HOG(1984), GIST(512), LBP(928) | This is the frequently used subsets of Caltech101 consisting of 20 categories of images built for object recognition tasks [[5]](#liMultiviewClusteringScalable2022). | Object         |
| Caltech101-all | 9144 | 102 | GABOR(48), WM(40), CENT(254), HOG(1984), GIST(512), LBP(928) | The Caltech101 dataset contains images from 101 object categories and **background** (e.g., “helicopter”, “elephant” and “chair” etc.) and a background category that contains the images not from the 101 object categories [[8]](#zhangBinaryMultiViewClustering2019). | Object         |
| CiteSeer | 3312 | 6 | Content(3703), Cites(4732) | The archive contains 3312 documents over the 6 labels (Agents,IR,DB,AI,HCI,ML) [[13]](#fangEfficientMultiviewClustering2023). | Text           |
| Cora | 2708 | 7 | Content(1433),  Inbound(2708), Outbound(2708), Cites(2708) | The archive contains 2708 documents over the 7 labels (Neural_Networks, Rule_Learning, Reinforcement_Learning, Probabilistic_Methods, Theory, Genetic_Algorithms,Case_Based) [[13]](#fangEfficientMultiviewClustering2023). | Text           |
| Handwritten | 2000 | 10 | FOU(76), FAC(216), KAR(64), PIX(240), ZER(47), MOR(6) | This dataset consists of features of handwritten numerals ('0'--'9') extracted from a collection of Dutch utility maps [[7]](#nieMultiviewClusteringAdaptively2018). | Image          |
| MNIST-10k | 10000 | 10 | ISO(30), LDA(9), NPE(30) | A freely available and well-known handwritten database for image recognition consisting of four categories from digit 0 to digit 9, and each category has 1,000 samples evenly [[1]](#dengMNISTDatabaseHandwritten2012). | Handwritten    |
| MNIST-4 | 4000 | 4 | ISO(30), LDA(9), NPE(30) | MNIST-4 is a subset of MNIST-10k consisting of four categories from digit 0 to digit 3 [[5]](#liMultiviewClusteringScalable2022). | Handwritten    |
| Movies | 617 | 17 | Keywords(1878), Actors(1398) | It is a movie corpus extracted from IMDb. [[22]](#xieGeneralizedMultiviewLearning2022) | Text           |
| MSRC-v5 | 210 | 7 | CM(24), HOG(576), GIST(512), LBP(256), CENT(254) | A subset of the Microsoft Research in Cambridge dataset [[21]](#xiaTensorizedBipartiteGraph2023). | Image          |
| NUS-WIDE-OBJ | 30000 | 31 | CH(65), CM(226), CORR(145), EDH(74), WT(129) | It is a dataset for object recognition which consists of 30000 images in 31 classes[[9]](#liLargescaleMultiviewSpectral2015). | Image          |
| NUS-WIDE | 2400 | 12 |  CH(64), CORR(144), EDH(75), WT(128), CM55(225) | 12 categories of animal images selected from the NUS-WIDE-OBJ dataset, and the first 200 images are selected for each category [[3]](#chuaNUSWIDERealworldWeb2009). | Object         |
| ORL | 400 | 40 | View1(4096), View2(3304), View3(6750) | Face dataset contains 400 images of 40 distinct subjects. For each category, images were taken at different times, lights, facial expressions (open / closed eyes, smiling or not) and facial details (with glasses / without glasses) [[23]](#luoConsistentSpecificMultiView2018) .| Face           |
| OutdoorScene | 2688 | 8 | GIST(512), HOG(432), LBP(256), Gabor(48) |  The dataset has 2688 outdoor scene images consisting of 8 groups [[13]](#fangEfficientMultiviewClustering2023). | Object         |
| ProteinFold | 694 | 27 | View1(27), View2(27), View3(27), View4(27), View5(27), View6(27), View7(27), View8(27), View9(27), View10(27), View11(27), View12(27) | Multiply kernel learning dataset on protein fold prediction [[19]](#liuEfficientEffectiveRegularized2021). | Protein        |
| Prokaryotic | 551 | 4 | gene-repert(393), proteome-comp(3), Text(438) | It contains prokaryotic species described with heterogeneous multi-view data including textual data and different genomic representations.[[20]](#niuMultiviewEnsembleClustering2023) | Genome         |
| Reuters-1200 | 1200 | 6 | English(2000), French(2000), German(2000), Spanish(2000), Italian(2000) | It contains 6 samples of 1200 documents over 6 labels, and desribed by 5 views of 2000 words each [[15]](#huangAutoweightedMultiviewClustering2020). | Text           |
| Reuters-1500 | 1500 | 6 | English(21531), French(24893), German(34279), Spanish(15506), Italian(11547) | The dataset is collection of 1500 documents which are expressed in five different languages (Italian, Spanish, French, German and English) and the corresponding translations [[14]](#weiFuzzyClusteringMultiview2022). | Text           |
| Reuters | 18758 | 6 | English(21531), French(24893), German (34279), Spanish (15506), Italian(11547) | This dataset consists of documents that are written in five different languages and their translations[[9]](#liLargescaleMultiviewSpectral2015). | Text           |
| UCI | 2000 | 10 | Intensity(240), FOU(76), MOR(6) | Obtained from the UCI machine learning repository, this dataset consists of 2000 handwritten digit images belonging to 10 digits with each digit containing 200 samples [[4]](#chenLowrankTensorBased2022). | Handwritten    |
| WebKB | 1051 | 2 | Anchor(1,840), Content(3,000) | The dataset for web pages collected from computer science department web sites at four universities: Cornell University, University of Washington,University of Wisconsin, and University of Texas [[6]](#qiangFastMultiviewDiscrete2021). | Text           |
| Wikipedia | 2866 | 10 | Word(128), SIFT(10) | Wikipedia dataset is the most widely-used dataset for cross-media retrieval. It is based on Wikipedia’s "featured articles", a continually updated article collection [[11]](#rasiwasiaNewApproachCrossmodal2010). | Text           |
| Wikipedia-test | 693 | 10 | Word(128), SIFT(10) | The test set of the Wikipedia collection [[10]](#shiFastMultiViewClustering2023). | Text           |
| Yale | 165 | 15 | Intensity(4096), LBP(3304), Gabor(6750) | This dataset consists of 165 gray-scale face images belonging to 15 subjects with each subject containing 11 images [[4]](#chenLowrankTensorBased2022). | Face           |

### List of features
- FOU: Fourier coefficients of the character shapes
- FAC: Profile correlations
- PIX: Pixel averages in 2 × 3 windows
- ZER: Zernike moment
- MOR: Morphological features.
- Gabor: Gabor feature
- WM: Wavelet moments
- CENT: CENTRIST feature [17]
- HOG: Histogram of oriented gradients feature
- GIST: GIST feature [18]
- LBP: Local binary patterns feature
- CH: Color Histogram
- TH: Texture Histogram
- CM: Color moments
- CS: Color similiarity
- CORR: Color correlation
- EDH: Edge distribution
- WT: Wavelet texture
- SD: Shape descriptor
- FSM: Fine scale margin
- SIFT: Scale Invariant Feature Transform

## References
<div id="dengMNISTDatabaseHandwritten2012"></div>

[1] L. Deng, “The MNIST Database of Handwritten Digit Images for Machine Learning Research [Best of the Web],”  _IEEE Signal Processing Magazine_, vol. 29, no. 6, pp. 141–142, Jan. 2012, doi: [10.1109/MSP.2012.2211477](https://doi.org/10.1109/MSP.2012.2211477).

<div id="winnLOCUSLearningObject2005"></div>

[2] J. Winn and N. Jojic, “LOCUS: learning object classes with unsupervised segmentation,” in _Tenth IEEE International Conference on Computer Vision (ICCV’05)_ Volume 1, Oct. 2005, pp. 756-763 Vol. 1. doi: [10.1109/ICCV.2005.148](https://doi.org/10.1109/ICCV.2005.148). 

<div id="wangRobustSelfWeightedMultiView2020"></div>

 [3] B. Wang, Y. Xiao, Z. Li, X. Wang, X. Chen, and D. Fang, “Robust Self-Weighted Multi-View Projection Clustering,” in _Proc. AAAI Conf. Artif. Intell._, Apr. 2020, pp. 6110–6117. doi: [10.1609/aaai.v34i04.6075](https://doi.org/10.1609/aaai.v34i04.6075).

<div id="chenLowrankTensorBased2022"></div>

[4] M.-S. Chen, C.-D. Wang, and J.-H. Lai, “Low-rank Tensor Based Proximity Learning for Multi-view Clustering,” _IEEE Transactions on Knowledge and Data Engineering_, pp. 1–1, Jan. 2022, doi: [10.1109/TKDE.2022.3151861](https://doi.org/10.1109/TKDE.2022.3151861).

<div id="liMultiviewClusteringScalable2022"></div>

[5] X. Li, H. Zhang, R. Wang, and F. Nie, “Multiview Clustering: A Scalable and Parameter-Free Bipartite Graph Fusion Method,” _IEEE Transactions on Pattern Analysis and Machine Intelligence_, vol. 44, no. 1, pp. 330–344, Jan. 2022, doi: [10.1109/TPAMI.2020.3011148](https://doi.org/10.1109/TPAMI.2020.3011148).

<div id="qiangFastMultiviewDiscrete2021"></div>

[6] Q. Qiang, B. Zhang, F. Wang, and F. Nie, “Fast Multi-view Discrete Clustering with Anchor Graphs,” in _Proc. AAAI Conf. Artif. Intell._, May 2021, pp. 9360–9367. doi: [10.1609/aaai.v35i11.17128](https://doi.org/10.1609/aaai.v35i11.17128).

<div id="nieMultiviewClusteringAdaptively2018"></div>

[7] F. Nie, L. Tian, and X. Li, “Multiview clustering via adaptively weighted procrustes,” in _Proc. ACM Int. Conf. Knowl. Discov. Data Min._, 2018, pp. 2022–2030. doi: [10.1145/3219819.3220049](https://doi.org/10.1145/3219819.3220049).

<div id="zhangBinaryMultiViewClustering2019"></div>

[8] Z. Zhang, L. Liu, F. Shen, H. T. Shen, and L. Shao, “Binary Multi-View Clustering,” _IEEE Transactions on Pattern Analysis and Machine Intelligence_, vol. 41, no. 7, pp. 1774–1782, Jul. 2019, doi: [10.1109/TPAMI.2018.2847335](https://doi.org/10.1109/TPAMI.2018.2847335).

<div id="liLargescaleMultiviewSpectral2015"></div>

[9]  Y. Li, F. Nie, H. Huang, and J. Huang, “Large-scale multi-view spectral clustering via bipartite graph,” in _Proc. AAAI Conf. Artif. Intell._, 2015. doi: [10.1609/aaai.v29i1.9598](https://doi.org/10.1609/aaai.v29i1.9598).

<div id="shiFastMultiViewClustering2023"></div>

[10]  S. Shi, F. Nie, R. Wang, and X. Li, “Fast Multi-View Clustering via Prototype Graph,” _IEEE Transactions on Knowledge and Data Engineering_, vol. 35, no. 1, pp. 443–455, Jan. 2023, doi: [10.1109/TKDE.2021.3078728](https://doi.org/10.1109/TKDE.2021.3078728).

<div id="rasiwasiaNewApproachCrossmodal2010"></div>

[11]  N. Rasiwasia et al., “A new approach to cross-modal multimedia retrieval,” in  _Proc. ACM Int. Conf. Multimedia_, 2010, pp. 251–260. doi: [10.1145/1873951.1873987](https://doi.org/10.1145/1873951.1873987).


<div id="lampertLearningDetectUnseen2009"></div>

[12] C. H. Lampert, H. Nickisch, and S. Harmeling, “Learning to detect unseen object classes by between-class attribute transfer,” in _Proc. IEEE Conf. Comput. Vis. Pattern Recognit._, Jun. 2009, pp. 951–958. doi: [10.1109/CVPR.2009.5206594](https://doi.org/10.1109/CVPR.2009.5206594).

<div id="fangEfficientMultiviewClustering2023"></div>

[13] S.-G. Fang, D. Huang, X.-S. Cai, C.-D. Wang, C. He, and Y. Tang, “Efficient Multi-view Clustering via Unified and Discrete Bipartite Graph Learning,” _IEEE Transactions on Neural Networks and Learning Systems_, pp. 1–12, Apr. 2023, doi: [10.1109/TNNLS.2023.3261460](https://doi.org/10.1109/TNNLS.2023.3261460).

<div id="weiFuzzyClusteringMultiview2022"></div>

[14] H. Wei, L. Chen, C. L. P. Chen, J. Duan, R. Han, and L. Guo, “Fuzzy clustering for multiview data by combining latent information,” _Applied Soft Computing_, vol. 126, p. 109140, Sep. 2022, doi: [10.1016/j.asoc.2022.109140](https://doi.org/10.1016/j.asoc.2022.109140).

<div id="huangAutoweightedMultiviewClustering2020"></div>

[15] S. Huang, Z. Kang, and Z. Xu, “Auto-weighted multi-view clustering via deep matrix decomposition,” _Pattern Recognition_, vol. 97, p. 107015, Jan. 2020, doi: [10.1016/j.patcog.2019.107015](https://doi.org/10.1016/j.patcog.2019.107015).

<div id="yangMultiViewAdjacencyConstrainedHierarchical2022"></div>

[16] J. Yang and C.-T. Lin, “Multi-View Adjacency-Constrained Hierarchical Clustering,” _IEEE Transactions on Emerging Topics in Computational Intelligence_, pp. 1–13, 2022, doi: [10.1109/TETCI.2022.3221491](https://doi.org/10.1109/TETCI.2022.3221491).

<div id="wuCentristVisualDescriptor2010"></div>

[17] J. Wu and J. M. Rehg, “Centrist: A visual descriptor for scene categorization,” _IEEE transactions on pattern analysis and machine intelligence_, vol. 33, no. 8, pp. 1489–1501, 2010.

<div id="olivaModelingShapeScene2001"></div>

[18] A. Oliva and A. Torralba, “Modeling the Shape of the Scene: A Holistic Representation of the Spatial Envelope,” _International Journal of Computer Vision_, vol. 42, no. 3, pp. 145–175, May 2001, doi: [10.1023/A:1011139631724](https://doi.org/10.1023/A:1011139631724).

<div id="liuEfficientEffectiveRegularized2021"></div>

[19] X. Liu _et al._, “Efficient and Effective Regularized Incomplete Multi-View Clustering,” _IEEE Transactions on Pattern Analysis and Machine Intelligence_, vol. 43, no. 8, pp. 2634–2646, Aug. 2021, doi: [10.1109/TPAMI.2020.2974828](https://doi.org/10.1109/TPAMI.2020.2974828).

<div id="niuMultiviewEnsembleClustering2023"></div>

[20] X. Niu, C. Zhang, X. Zhao, L. Hu, and J. Zhang, “A multi-view ensemble clustering approach using joint affinity matrix,” _Expert Systems with Applications_, vol. 216, p. 119484, Apr. 2023, doi: [10.1016/j.eswa.2022.119484](https://doi.org/10.1016/j.eswa.2022.119484).

<div id="xiaTensorizedBipartiteGraph2023"></div>

[21] W. Xia, Q. Gao, Q. Wang, X. Gao, C. Ding, and D. Tao, “Tensorized Bipartite Graph Learning for Multi-View Clustering,” _IEEE Transactions on Pattern Analysis and Machine Intelligence_, vol. 45, no. 4, pp. 5187–5202, Apr. 2023, doi: [10.1109/TPAMI.2022.3187976](https://doi.org/10.1109/TPAMI.2022.3187976).

<div id="xieGeneralizedMultiviewLearning2022"></div>

[22] X. Xie and Y. Xiong, “Generalized multi-view learning based on generalized eigenvalues proximal support vector machines,” _Expert Systems with Applications_, vol. 194, p. 116491, May 2022, doi: [10.1016/j.eswa.2021.116491](https://doi.org/10.1016/j.eswa.2021.116491).

<div id="luoConsistentSpecificMultiView2018"></div>

[23] S. Luo, C. Zhang, W. Zhang, and X. Cao, “Consistent and Specific Multi-View Subspace Clustering,” in _Proc. AAAI Conf. Artif. Intell._, Apr. 2018. doi: [10.1609/aaai.v32i1.11617](https://doi.org/10.1609/aaai.v32i1.11617).

<div id="wuEssentialTensorLearning2019"></div>

[24] J. Wu, Z. Lin, and H. Zha, “Essential Tensor Learning for Multi-View Spectral Clustering,” _IEEE Transactions on Image Processing_, vol. 28, no. 12, pp. 5910–5922, Feb. 2019, doi: [10.1109/TIP.2019.2916740](https://doi.org/10.1109/TIP.2019.2916740).

<div id="zhangFacilitatedLowrankMultiview2023"></div>

[25] G.-Y. Zhang, D. Huang, and C.-D. Wang, “Facilitated low-rank multi-view subspace clustering,” _Knowledge-Based Systems_, vol. 260, p. 110141, Jan. 2023, doi: [10.1016/j.knosys.2022.110141](https://doi.org/10.1016/j.knosys.2022.110141).

