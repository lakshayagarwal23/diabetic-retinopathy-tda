Multi-Class Grading of Diabetic Retinopathy Severity

Diabetic Retinopathy (DR) is a progressive complication of diabetes in
which high blood sugar damages the retina's blood vessels, leading to
microaneurysms, hemorrhages, neovascularization, and eventually
blindness. Approximately 830 million people worldwide have diabetes, and
80% of them are at risk of developing DR. Early stages of DR are
asymptomatic, meaning retinal damage begins long before the patient
notices any change in vision. This makes early-stage detection critical,
prevents irreversible blindness, detects retinal damage before vision
loss begins, and improves long-term treatment outcomes \[1, 2\].

Screening programs typically rely on color fundus photography, a
low-cost and non-invasive imaging modality. These images contain a
variety of diagnostic information such as vessel structures, lesion
patterns, retinal texture, but the subtle lesions in early-stage DR
occupy a very small portion of the image and require expert
interpretation. Manual grading performed by ophthalmologists is reliable
but suffers from major limitations: it is slow, labor-intensive, and
difficult to scale. Large-scale screening programs require consistent
severity grading across thousands or millions of patients, which is not
feasible with purely manual workflows. The challenge extends beyond
classification as models must learn to capture both local patterns and
global retinal structure. This highlights why DR grading is a difficult
task that requires deep learning. But, automated DR grading is faster,
consistent and helps as an objective assessment \[3\].

A wide range of computational approaches have been explored for
automated DR grading. These can be broadly divided into three
categories: traditional CNN-based classifiers, hybrid CNN models with
attention or multiple branches, and topology-informed approaches. The
progression of methods can be viewed across three major phases:
foundation, performance optimization, and specialization.

**Foundation Phase (early 2010s):** Early DR detection systems were
dominated by single CNN models such as AlexNet, VGG, DenseNet, and
ResNet. These architectures brought powerful representation learning to
medical imaging, but initially struggled with the nuances of DR grading.

- Accuracy plateaued around 50--52% for 5-class DR grading \[4\].

- CNNs often failed to capture small subtle lesions.

- Variability in image illumination and contrast reduced model
  reliability.

The main breakthroughs at this time include: 1) Transfer learning and 2)
Image preprocessing techniques such as CLAHE and BenGraham
normalization.

**Performance Phase (late 2010s):** As the field grew, the focus went on
improving accuracy and robustness. Here, two major classes of models
came up: 1) Ensemble Models and 2) Attention-based models. Ensembles
combining EfficientNet or similar architectures became the dominant
strategy in Kaggle competitions and academic benchmarks. SOTA models
achieved up to 85% accuracy in 5-class DR grading \[5\]. However, there
were high computational costs and the models were not clinically
interpretable due to attention masks.

**Specialisation Phase (Current Focus):** The recent work in the field
has moved towards integrating higher-level structure and global retina
geometry. Topology-informed models emerged, harnessing topological data
analysis (TDA) to compute the global shape and connectivity information
for vessel patterns, microaneurysms, and connected components. TDA
techniques such as persistence diagrams and persistence images provide
structural insights that CNNs often overlook. Transformers have also
recently been adapted for DR detection due to their ability to model
long-range relationships. They excel at capturing global retinal
structure and integrating contextual cues over large receptive fields.
However, transformers require large datasets and are computationally
very expensive.

Ahmed (2025) presents a comprehensive evaluation of pretrained CNN
architectures, including variants of ResNet and EfficientNet, fine-tuned
on the APTOS-2019 retinal fundus dataset for both binary and multi-class
DR severity grading \[6\]. The study introduces a class-balanced
augmentation pipeline that expands minority classes to 20,000 samples
each, effectively mitigating the significant class imbalance inherent in
DR datasets. This augmentation strategy produced more stable and
consistent performance across DR stages.

The results demonstrate that EfficientNet variants outperform deeper
ResNet models, achieving 98.9% accuracy (AUC 99.4%) for binary
classification and 84.6% accuracy (AUC 94.1%) for five-class DR grading.
The findings highlight that representational efficiency drives
generalization when trained on limited clinical data. The study also
establishes a reproducible DR benchmark by coupling transfer learning
with balanced augmentation, enabling more scalable and computationally
efficient screening models.

Moving forward, Ahmed, Bhuiyan, and Coskuner (2025) propose Topo-Net, a
hybrid deep learning architecture that integrates Topological Data
Analysis (TDA), specifically persistent homology, with conventional CNN
embeddings \[7\]. The model extracts Betti-based topological features
from retinal images, capturing structural properties such as vessel
connectivity, lesion shape, and regional geometry. These features are
fused with CNN-derived representations to generate a more holistic
encoding of retinal pathology.

Topo-Net exhibited strong generalization across multiple ophthalmic
datasets, including APTOS (AUC 95.0%, accuracy 80%), OCT-HAMD (AUC
93.6%), and ORIGA (AUC 83.0%), surpassing baseline CNN models in DR. The
study demonstrates that TDA provides discriminative global features that
complement local pixel-based CNN filters.

**Dataset**

This study utilizes the APTOS 2019 Blindness Detection dataset, provided
by the Asia Pacific Tele-Ophthalmology Society (APTOS) in collaboration
with Aravind Eye Hospital (India). The dataset contains 3,662 color
fundus photographs, each labeled according to a five-stage diabetic
retinopathy (DR) severity scale:

  ------------------------------------------------------------------------
  Class       Description                 Number of Images
  ----------- --------------------------- --------------------------------
  0           No DR                       1,805

  1           Mild DR                     370

  2           Moderate DR                 999

  3           Severe DR                   193

  4           Proliferative DR            295
  ------------------------------------------------------------------------

### *Aptos 2019 Dataset Class Distribution*

### **Challenges**

The dataset presents two significant challenges:

1.  Severe class imbalance\
    The ratio between the largest and smallest classes (No_DR vs.
    Severe) is approximately 9.4:1, which can bias models toward
    majority classes.

2.  High variability in image resolution\
    Image sizes range from 474×358 to 3388×2588, requiring careful
    preprocessing for consistent downstream analysis.

**Preprocessing Pipeline**

a.  Load and Resize

- Raw RGB fundus images are loaded from disk.

- Each image is uniformly resized to 256×256 pixels to ensure consistent
  input dimensions for the CNN.

b.  Green Channel Extraction

- Only the green channel from the RGB image is retained, producing a
  256×256 grayscale map.

- The green spectrum is preferred in ophthalmic imaging because it
  maximizes the contrast of vessels, microaneurysms, and hemorrhages by
  suppressing orange-red illumination artifacts \[8\].

c.  Contrast Enhancement (Histogram Equalization)

- A CLAHE-like histogram equalization step is applied to the green
  channel.

- Equalization redistributes intensity values, enhancing the visibility
  of early-stage lesions and vessel boundaries.

- Before-and-after histograms show improved uniformity and increased
  dynamic range.

d.  Circular Retina Masking

- A circular mask centered on the image is used to isolate the retinal
  region.

- All pixels outside the circular boundary are zeroed out to remove
  irrelevant black corners and reduce noise in topological computations.

e.  Intensity Normalization

- Two normalization steps are applied within the retinal mask:

1.  Z-score normalization\
    ![](./media/media/image4.png){width="1.5416666666666667in"
    height="0.6770833333333334in"}\
    This standardizes illumination variations across images.

2.  Min--Max scaling to \[0, 1\]

> ![](./media/media/image8.png){width="1.806700568678915in"
> height="0.680663823272091in"}\
> Required for numerical stability in persistent homology calculations.

f.  Topology Construction: Once preprocessed, the normalized
    green-channel image is used to construct the topological
    representation.

- Cubical Complex Formation: A Gudhi Cubical Complex is constructed from
  the 2D intensity surface. This enables computation of persistent
  homology on the image domain.

- Persistent Homology: Persistent homology tracks the "birth" and
  "death" of these features over a filtration induced by pixel
  intensities.

  - H₀ (0-dimensional): connected components

  - H₁ (1-dimensional): loops, rings, circular structures\
    (relevant for microaneurysms and vascular patterns)

<!-- -->

- Persistence Diagrams: Each topological feature is plotted as a point
  (bi,di), representing its birth and death times. Points far from the
  diagonal correspond to more persistent, and potentially more
  meaningful, retinal structures.

- Betti Curves:Betti curves record the number of active topological
  features as a function of threshold ttt. These curves summarize the
  evolution of retinal structure across intensity levels.

- Persistence Image: Finally, the persistence diagram is converted into
  a 64×64 persistence image:

  - The diagram is rasterized into a fixed grid.

  - Gaussian kernels "smooth" feature contributions into a continuous 2D
    heatmap.

  - This persistence image becomes the TDA input to the CNN's
    topology-aware attention module.

![](./media/media/image9.png){width="6.364583333333333in"
height="3.3229166666666665in"}

*Preprocessing Pipeline*

**Architecture**

![](./media/media/image11.png){width="6.5in"
height="3.638888888888889in"}

*Proposed Architecture*

The proposed framework comprises two synergistic branches: a
**Convolutional Neural Network (CNN) branch** for visual feature
extraction and a **Topological Data Analysis (TDA) branch** for
structural reasoning. Together, these modules aim to enhance both the
**accuracy** and **interpretability** of diabetic retinopathy (DR)
grading by integrating intensity-based and topology-aware information.

### **CNN Branch --- Visual Feature Extraction**

The upper branch of the architecture processes the RGB fundus image,
resized to **3×256×256**, using a **pre-trained EfficientNet-B0**
backbone. The network consists of an initial **stem convolution** layer
followed by multiple **mobile inverted bottleneck (MBConv) blocks** with
ReLU activations.\
These layers progressively capture both local and global retinal
patterns---such as blood vessel structure, microaneurysms, and optic
disc textures---resulting in a dense feature representation denoted as:

![](./media/media/image7.png){width="1.0052088801399826in"
height="0.2571467629046369in"}

For EfficientNet-B0, **C = 1280** and **H = W = 8**. This feature tensor
Fi serves as a compact encoding of the retinal image, preserving
essential spatial semantics while reducing dimensionality.

### **TDA Branch --- Topological Feature Extraction**

In parallel, the lower branch performs **topological analysis** on the
**green channel** of the same fundus image, which provides the highest
vessel-to-background contrast.\
This channel undergoes a series of preprocessing steps including
**histogram equalization**, **circular masking**, and **normalization**
to suppress peripheral noise and emphasize vascular connectivity.

Subsequently, a **persistence diagram (PD)** is computed using *Cubical
Complexes* to characterize the topological features of the
image---namely, connected components (H₀) and loops (H₁).\
The persistence diagram is then transformed into a **persistence image
(PI)**, a fixed-size 2D grid representation:

![](./media/media/image2.png){width="0.9947922134733158in"
height="0.21212489063867015in"}

This persistence image acts as a **topological mask**, encoding
geometric relationships among vessels and lesions.\
It is processed through the **Topology-Guided Spatial Attention (TGSA)**
module, which applies **max pooling**, **average pooling**, and **1×1
convolution** followed by a **sigmoid activation** to generate a **soft
attention map** aligned with the CNN feature space.

### **TGSA Integration --- Attention Fusion**

The soft attention map from the TDA branch is **element-wise
multiplied** with the CNN feature map FiFi​ to generate topologically
refined representations:

![](./media/media/image6.png){width="2.119792213473316in"
height="0.40889435695538057in"}

This operation reweights the CNN features according to the topological
significance of regions within the image, ensuring that the model
focuses on diagnostically relevant structures---such as vessel loops,
hemorrhages, and microaneurysms---while suppressing irrelevant
high-contrast areas like the optic disc.

### **Loss Functions**

The model is trained using a **dual-objective loss function** that
balances classification performance with topological alignment:

![](./media/media/image5.png){width="1.7239588801399826in"
height="0.3038123359580053in"}

where

- LCE​ is the **Cross-Entropy Loss** used for multi-class DR grade
  classification, and

- Lmask​ is the **Mean Squared Error (MSE)** between the CNN-generated
  attention map and the TDA-derived topological mask.

The second term encourages the CNN to align its focus with clinically
meaningful topological regions. The weighting factor λ=1.0 ensures both
objectives contribute equally to learning.

### **Final Classification**

The topology-refined feature map Fi′ is passed through a
**post-convolutional block**, followed by **global average pooling** and
a **fully connected layer**.\
A **softmax layer** outputs the final probability distribution across
the **five DR grades (0--4)**, representing increasing disease severity.

**Results**

The performance of the proposed topology-guided CNN model was evaluated
on the APTOS-2019 test set and compared against Topo-Net. Results
demonstrate that the proposed model achieves substantially higher
accuracy and improved agreement with ground-truth DR severity labels.

Topo-Net (2025) reports an overall test accuracy of **80%**, with
**82.2% precision** and **78.2% recall** for multi-class DR grading.
While Topo-Net effectively incorporates topological priors using
persistent homology, its performance remains constrained by post-hoc
topological fusion and limited attention guidance.

In contrast, the proposed model, integrating topology as a spatial
attention prior rather than as auxiliary features, achieves
significantly stronger results. On the APTOS test set, the model
records:

- Accuracy: 86.43%

- AUC: 0.92

- Quadratic Weighted Kappa (QWK): 0.9236

The QWK score, a gold-standard metric for ordinal DR classification,
indicates very high agreement between predicted and true severity
grades, outperforming both traditional CNNs and prior TDA-based
approaches.

![](./media/media/image10.png){width="2.9996434820647417in"
height="1.012866360454943in"}

*Proposed Architecture Test Metrics*

The model performs best on the "No_DR" and "Moderate" classes, which
constitute the largest subsets of the dataset. Mild DR shows moderate
recall but lower precision, reflecting the inherent difficulty of
detecting small early-stage lesions. Severe and proliferative DR classes
exhibit stronger precision than recall, consistent with their lower
representation in the dataset.

Overall, the model demonstrates stable class-wise behavior with
misclassifications primarily occurring between adjacent DR stages, an
expected pattern given the ordinal nature of the grading system. The
proposed topology-guided attention model surpasses prior
state-of-the-art topology-based approaches by a considerable margin,
particularly in QWK and overall accuracy. The integration of persistent
homology as a spatial attention prior appears to enhance lesion
localization and improve ordinal classification reliability.

**Novelty:**

This work introduces three key innovations that differentiate it from
prior diabetic retinopathy (DR) classification models.

**Topology as a Spatial Attention Prior**

Previous topology-driven models, including Topo-Net, incorporate
persistent homology outputs as **post-hoc auxiliary features**
concatenated with CNN embeddings. While effective, this approach does
not influence where the CNN directs its attention during feature
extraction. In contrast, the proposed method introduces a
**topology-conditioned spatial attention mechanism**, in which the
persistence image directly modulates convolutional activations:

![](./media/media/image1.png){width="3.7656255468066493in"
height="0.34375in"}

This design enables topological information---such as loops, rings, and
vascular structures---to act as a **prior** that guides the CNN's focus
to structurally meaningful retinal regions. Rather than serving as an
auxiliary descriptor, topology becomes an integral, learned component of
the feature extraction process.

**Green Channel Preprocessing**

Use of green-channel--based preprocessing to enhance early-stage lesions
prior to both TDA and CNN analysis. Ophthalmologists routinely employ
green-filtered imaging to improve visibility of microaneurysms and
hemorrhages by attenuating red light. Integrating this principle
computationally improves: lesion contrast,vessel visibility, and
stability of topological filtrations. This domain-aligned enhancement
provides a clearer structural representation, strengthening both
topological and deep learning components \[9\].

**Mixup Data Augmentation**

Mixup is incorporated to mitigate dataset imbalance and overfitting. New
training samples are generated via convex combinations of image pairs
and their labels:

![](./media/media/image3.png){width="6.5in"
height="0.5555555555555556in"}

**Conclusion**

This work presents a hybrid TDA--CNN framework for diabetic retinopathy
classification that integrates domain-informed preprocessing,
topological feature extraction, and topology-guided spatial attention.
By leveraging the green channel to enhance lesion visibility, using
persistent homology to capture global retinal structure, and employing
Mixup to mitigate class imbalance, the proposed model achieves robust
and interpretable multi-class DR grading.

The approach outperforms prior topology-based methods such as Topo-Net,
demonstrating higher accuracy and substantially improved quadratic
weighted kappa, reflecting stronger agreement with clinical severity
labels. These results highlight the value of combining structural
topology with deep visual representations and suggest that
topology-guided attention mechanisms offer a promising pathway toward
more reliable and clinically aligned DR screening systems.

**Bibliography:**

\[1\] Taher, Muhammad. \"The challenges of using Escherichia coli as a
host in recombinant insulin production. \" Journal of Pharmacy 5, no. 1
(2025): 1-5.

\[2\] "Eyes on Diabetes." WHO EMRO (2016),
[[https://www.emro.who.int/noncommunicable-diseases/highlights/eyes-on-diabetes.html]{.underline}](https://www.emro.who.int/noncommunicable-diseases/highlights/eyes-on-diabetes.html)

\[3\] Nderitu, Paul, Joan M. Nunez do Rio, Ms Laura Webster, Samantha S.
Mann, David Hopkins, M. Jorge Cardoso, Marc Modat, Christos Bergeles,
and Timothy L. Jackson.

"Automated image curation in diabetic retinopathy screening using deep
learning." Scientific Reports 12, no. 1 (2022): 11196.

\[4\] [Lam, Carson, Darvin Yi, Margaret Guo, and Tony Lindsey.
\"Automated detection of diabetic retinopathy using deep
learning.\"]{.mark} *AMIA summits on translational science
proceedings*[2018 (2018): 147.]{.mark}

\[5\] Chilukoti, Sai Venkatesh, Liqun Shan, Vijay Srinivas Tida, Anthony
S. Maida, and Xiali Hei.

"A reliable diabetic retinopathy grading via transfer learning and
ensemble learning with

quadratic weighted kappa metric." BMC Medical Informatics and Decision
Making 24, no. 1 (2024): 37.

[\[6\] Ahmed, Faisal.\"Addressing High Class Imbalance in Multi-Class
Diabetic Retinopathy Severity Grading with Augmentation and Transfer
Learning.\" arXiv preprint arXiv:2507.17121(2025).]{.mark}

[\[7\] Ahmed, Faisal, Mohammad Alfrad Nobel Bhuiyan, and Baris
Coskunuzer. \"Topo-CNN: Retinal Image Analysis with Topological Deep
Learning.\" Journal of Imaging Informatics in Medicine (2025):
1-17.]{.mark}

[\[8\]Castro, Mac Gayver da Silva, Francisco Vagnaldo Fechine Jamacaru,
Manoel Odorico de Moraes Filho, Paulo Roberto Leitão de Vasconcelos, and
Conceição Aparecida Dornelas.\"Enhanced performance in automated
diabetic retinopathy diagnosis achieved through Voronoi diagrams and
artificial intelligence.\" Scientific Reports 15, no. 1 (2025):
35763.]{.mark}

[\[9\] Moon, J. Y., K. M. Wai, N. S. Patel, R. Katz, M. Dahrouj, and J.
B. Miller. "Visualization of Retinal Breaks on Ultra-Widefield Fundus
Imaging Using a Digital Green Filter." Graefe's Archive for Clinical and
Experimental Ophthalmology 261, no. 4 (2023): 935--40.
https://doi.org/10.1007/s00417-022-05855-8.]{.mark}
