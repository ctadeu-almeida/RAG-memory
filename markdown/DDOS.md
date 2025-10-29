DDOS: The Drone Depth and Obstacle Segmentation Dataset

Benedikt Kolbeinsson
Imperial College London
bk915@imperial.ac.uk

Krystian Mikolajczyk
Imperial College London
k.mikolajczyk@imperial.ac.uk

https://huggingface.co/datasets/benediktkol/DDOS

Abstract

The advancement of autonomous drones, essential for
sectors such as remote sensing and emergency services,
is hindered by the absence of training datasets that fully
capture the environmental challenges present in real-world
scenarios, particularly operations in non-optimal weather
conditions and the detection of thin structures like wires.
We present the Drone Depth and Obstacle Segmentation
(DDOS) dataset to ﬁll this critical gap with a collection of
synthetic aerial images, created to provide comprehensive
training samples for semantic segmentation and depth esti-
mation. Speciﬁcally designed to enhance the identiﬁcation
of thin structures, DDOS allows drones to navigate a wide
range of weather conditions, signiﬁcantly elevating drone
training and operational safety. Additionally, this work in-
troduces innovative drone-speciﬁc metrics aimed at reﬁn-
ing the evaluation of algorithms in depth estimation, with
a focus on thin structure detection. These contributions
not only pave the way for substantial improvements in au-
tonomous drone technology but also set a new benchmark
for future research, opening avenues for further advance-
ments in drone navigation and safety.

1. Introduction

Fully autonomous drones are poised to revolutionize a mul-
titude of sectors, including remote sensing [3, 16, 17, 26,
31, 35], package delivery [4, 13], emergency services, and
disaster response [2, 9–11, 28, 29]. While manually con-
trolled drones have been effectively employed in speciﬁc
sectors, the advent of fully autonomous drones is poised
to unlock an array of novel applications, enhancing efﬁ-
ciency and expanding capabilities. However, realizing this
potential is contingent upon the ability of drones to navi-
gate safely and autonomously, which in turn requires a pre-
cise understanding of their environment. Current datasets
for training drone navigation systems are inadequate, par-
ticularly in representing challenging scenarios such as the
detection of thin structures like wires and cables, and oper-
ation under diverse weather conditions [25]. This deﬁciency

highlights the need for a dataset that provides a comprehen-
sive representation of the environment, enabling accurate
semantic segmentation and depth estimation across a wide
range of objects and conditions.

To address this gap, we introduce the Drone Depth
and Obstacle Segmentation (DDOS) dataset, a novel re-
source designed to signiﬁcantly enhance the training of au-
tonomous drones. DDOS stands out for its dual empha-
sis on depth and semantic segmentation annotations, with a
particular focus on the precise identiﬁcation of thin struc-
tures (a critical but often overlooked aspect in existing
datasets). By incorporating advanced computer graphics
and rendering techniques, DDOS generates synthetic aerial
images that mirror the complexity of real-world environ-
ments, encompassing a variety of settings and weather con-
ditions ranging from clear skies to adverse weather scenar-
ios such as rain, fog, and snowstorms.

Our objectives with the DDOS dataset are twofold:
ﬁrstly, to provide a richly annotated resource that reﬂects
the diversity of scenarios encountered by drones, with a par-
ticular focus on thin structures and adverse weather condi-
tions. Secondly, to enable the development and evaluation
of algorithms that signiﬁcantly improve the safety, reliabil-
ity, and operational efﬁciency of autonomous drones. By
achieving these objectives, we aim to bridge the gap in ex-
isting datasets and facilitate the advancement of drone tech-
nology to meet the demands of real-world applications.

We present a thorough analysis of DDOS which explores
key characteristics including class density, ﬂight dynamics,
and spatial distribution, providing a granular understanding
of its composition and capabilities. Through comparative
analysis with existing datasets, we highlight DDOS’s con-
tributions such as incorporating numerous thin and ultra-
thin structures with accurate depth and segmentation la-
bels, as well as diverse weather conditions. Furthermore,
we propose new drone-speciﬁc metrics designed to accu-
rately evaluate class-speciﬁc depth estimation performance.
These metrics are tailored to reﬂect the operational realities
of drone applications, offering a reﬁned lens through which
to assess algorithmic performance and contributing to the
broader goal of advancing drone technology and safety.

ThisCVPRWorkshoppaperistheOpenAccessversion,providedbytheComputerVisionFoundation.Exceptforthiswatermark,itisidenticaltotheacceptedversion;thefinalpublishedversionoftheproceedingsisavailableonIEEEXplore.7328AeroScapes Ruralscapes Mid-Air

Data type

Flight Trajectories
Frames
Labeled frames

Resolution
Frame rate
Environment
Camera motion
Altitude

Weather variations
Camera pose
Optical ﬂow
Depth map
Segmentation
Thin structures
Mesh structures

USF
[7]
Real

86
6 k
3 k

NE-VBWD
[33]
Real

TTPLA
[1]
Real

41
15 k
91

80
1 k
1 k

640

480

⇥
25 Hz
Town
Handheld
2 m

6576

4384

⇥
2 Hz
Town/Nature
Helicopter
+300 m

3840

2160

⇥
30 Hz
Pylons
Drone
-

No
No
No
No

No
No
No
Sparse

Wires only Wires only

Yes
No

Yes
No

No
No
No
No
Yes
Yes
Rough

PIM
[36]
Real

NA
159
159

960

1280
⇥
-
Pylons
Drone
-

No
No
No
No
No
Patches
Patches

UAVid
[21]
Real

30
300
300

3840

2160

⇥
0.2 Hz
Town/Nature
Drone
50 m

No
No
No
No
Yes
No
No

[27]
Real

141
3 k
3 k

720

1280
⇥
-
Various
Drone
5 – 50 m

No
No
No
No
Yes
Yes
Large only

[23]
Real

20
51 k
1 k*

[12]
Synthetic

54
119 k†
119 k†

TartanAir
[38]
Synthetic

SynthWires
[22]
Synthetic

SynDrone
[30]
Synthetic

DDOS
(ours)
Synthetic

1037
1 M
1 M

154
68 k
68 k

8
72 k
72 k

340
34 k
34 k

3840

2160

⇥
50 Hz
Town/Nature
Drone
-

1382

512

⇥
25 Hz
Nature
Drone
-

640

480

⇥
-
Various
Random
-

640

480

⇥
-
Various
Drone
-

1920

1080

⇥
25 Hz
Town
Drone
20, 50, 80 m

1280

720

⇥
10 Hz
Town/Nature
Drone
1 – 25 m

No
No
No
No
Yes
No
No

Yes
Yes
No
Yes
Yes
No
No

No
Yes
Yes
Yes
No‡
No‡
No‡

No
No
No
No
Wires only
Yes
No

No
Yes
No
Yes
Yes
No
No

Yes
Yes
Yes
Yes
Yes
Yes
Yes

Table 1. Comparison between our DDOS dataset and related datasets. *Ruralscapes also includes automatically generated labels for
the remaining 98% of the dataset. †Mid-Air includes additional variations for the same trajectory. ‡TartanAir does not include labeled
segmentation classes (i.e. each object is assigned to a random unlabeled class, with variations of the same object type in different classes).

Finally, we present baseline results obtained by applying
state-of-the-art algorithms to the DDOS dataset, establish-
ing a benchmark for future research in thin object detec-
tion. We examine the strengths and limitations of current
methodologies, particularly highlighting their notable fail-
ure to accurately predict the depth of thin structures. This
analysis emphasizes signiﬁcant opportunities for reﬁnement
and innovation within this domain.

To summarize, our main contributions are:

• DDOS Dataset: We present the Drone Depth and Obsta-
cle Segmentation (DDOS) dataset, a comprehensive re-
source developed to signiﬁcantly improve the training of
autonomous drones through extensive depth and semantic
segmentation annotations, with a special focus on accu-
rately identifying thin structures.

• Statistical Analysis and Dataset Comparison: We pro-
vide a thorough examination of the DDOS dataset, high-
lighting its unique attributes such as class distributions,
spatial distribution, and ﬂight dynamics. Our analysis
is enriched by a detailed comparative study, positioning
DDOS in the broader context of existing datasets and
underscoring its distinctive value in addressing speciﬁc
challenges in drone navigation.

• Drone-Speciﬁc Metrics: Novel drone-speciﬁc metrics
are introduced, tailored to the nuances of drone appli-
cations, particularly in the evaluation of depth accuracy.
These metrics offer a reﬁned and specialized framework
for assessing algorithmic performance.

• Baseline Results and Discussion: We present baseline
results from applying state-of-the-art algorithms to the
DDOS dataset, establishing benchmarks for thin object
detection research. Our discussion identiﬁes a critical
shortfall in existing depth estimation methods, emphasiz-
ing the need for future advancements.

2. Related Work

The scarcity of high-quality drone datasets hampers au-
tonomous drone training. This section reviews relevant
datasets, evaluating their strengths and weaknesses in re-
gards to training autonomous drones. These evaluations are
summarized in Table 1.

2.1. Driving datasets

The KITTI [15, 24], Cityscapes [8], nuScenes [6], and
Waymo [34] datasets, essential in computer vision for au-
tonomous driving, fall short in addressing drone-speciﬁc
requirements. KITTI’s concentration on road scenes lacks
the aerial views and diverse thin structures crucial for drone
navigation. Similarly, Cityscapes, nuScenes, and Waymo
fail to capture the unique aerial perspectives and the slen-
der objects like wires and cables vital for drone safety. The
absence of these aerial viewpoints and the limited represen-
tation of thin structures mean that models trained on these
datasets are not fully equipped to meet the challenges of
drone-based navigation.

2.2. Wire detection datasets

Several datasets have been speciﬁcally designed to tackle
the challenge of wire detection, given its critical importance
for ensuring the safety of low-ﬂying drones.

The USF dataset [7] and NE-VBWD [33] are pivotal re-
sources dedicated to wire detection, offering a unique per-
spective on the challenges of identifying thin structures in
aerial imagery. The USF dataset, while extensive, is limited
by its image quality and the accuracy of its wire annotations,
which are not pixel-accurate and often overlook the real-
world curvature of wires, instead deﬁning them as straight
lines. This simpliﬁcation fails to capture the complexity
of wire shapes in various environments, undermining the
dataset’s utility for training models to detect thin structures

7329accurately. NE-VBWD, although a more recent addition,
offers pixel-wise annotations and distance information, fo-
cusing on long-range wire detection. However, its suitabil-
ity for drone applications is limited due to its emphasis on
wires located at distances more relevant to manned aircraft,
thus diminishing its relevance for low-altitude drone opera-
tions where proximity to wires is a critical safety concern.
TTPLA [1] and PIM [36] also contribute to the ﬁeld
by focusing on transmission towers and power lines, with
TTPLA utilizing drone imagery but lacking depth infor-
mation, and PIM providing small image patches for wire
detection without offering semantic segmentation. These
datasets, while enriching the domain with speciﬁc insights
into wire and tower detection, similarly fall short in address-
ing the broad needs of autonomous drone navigation, such
as a diverse range of thin structures, depth mapping, and en-
vironmental conditions beyond the mere presence of wires.

2.3. Drone datasets

UAVid [21], AeroScapes [27], and Ruralscapes [23] serve
as general drone datasets. They provide a broader view of
urban and rural landscapes from a drone’s perspective, in-
cluding various object classes for semantic segmentation.
Despite their wider scope, these datasets still lack sufﬁcient
emphasis on thin structures, such as wires, which are crucial
for the safe navigation of drones in complex environments.
SynthWires [22] utilizes a different approach by over-
laying synthetic wires over real-world images from drones.
This method enhances the variety of wire scenarios avail-
able for training, although the absence of depth information
limits the dataset’s applicability for comprehensive 3D nav-
igation and obstacle avoidance training.

In enhancing the dataset landscape for drone navigation
research, Mid-Air [12], TartanAir [38], and SynDrone [30]
represent signiﬁcant contributions as synthetic datasets of-
fering voluminous labeled training samples. These datasets
play a pivotal role in simulating a diverse array of ﬂight dy-
namics and environmental conditions, providing essential
assets such as precise depth maps and camera poses critical
for the advancement of sophisticated drone navigation al-
gorithms. Despite their value, these datasets exhibit certain
limitations that restrict their comprehensive utility in fully
leveraging the potential of synthetic data generation.

One notable shortfall is their failure to encapsulate a
complete spectrum of ﬂight scenarios, particularly those in-
volving close encounters, aggressive maneuvering, and very
low-altitude ﬂying. Such scenarios, while perilous to exe-
cute in real-world settings, are quintessential for preparing
drones to navigate through complex, unpredictable environ-
ments. Synthetic datasets, with their capacity for controlled
simulation, are uniquely positioned to safely incorporate
these high-risk ﬂight patterns, thereby enriching the train-
ing regime without endangering equipment or safety.

Moreover, while synthetic datasets offer the advantage
of generating pixel-perfect segmentation and precise depth
measurements, especially for thin structures – attributes
unattainable with conventional data collection methods –
they fall short in representing thin structures like wires, ca-
bles, and fences. These elements are critical for ensuring
the navigational reliability of drones in densely populated
or structurally complex areas. The absence of such objects
in the datasets underscores a missed opportunity to leverage
some of the beneﬁts of synthetic data generation.

Our proposed dataset, DDOS, is designed to surpass
the limitations of existing datasets in wire detection and
drone navigation.
It provides detailed representations of
thin structures and a wide array of other entities, incorpo-
rating weather variability and extensive drone motion. Its
synthetic foundation enables simulations of close encoun-
ters with objects, typically unsafe in reality, enhancing the
dataset’s utility and realism for drone training.

3. Dataset Features

We introduce the DDOS dataset, speciﬁcally designed for
the training of autonomous drones, utilizing synthetic data
generation to compile 340 unique drone ﬂights. This dataset
is characterized by its comprehensive coverage of various
weather conditions, from clear skies to snowstorms, and in-
cludes high-risk scenarios such as close encounters and mi-
nor collisions. These scenarios, crucial for drone training,
are typically too hazardous to replicate in real-world set-
tings. The dataset is notable for its provision of pixel-level
precision in semantic segmentation and depth information,
particularly for challenging objects such as wires, cables,
and fences, thus offering a photo-realistic simulation of en-
vironments drones are likely to encounter.

Each ﬂight within the DDOS dataset consists of 100
frames, culminating in a total of 34 000 frames across the
dataset. This substantial volume of data supports detailed
analysis and algorithm training. The dataset emphasizes
thin structures, which present signiﬁcant navigational chal-
lenges, thereby serving as a critical resource for the de-
velopment of algorithms that require precise segmentation
and depth estimation capabilities in complex aerial scenar-
ios. Accompanying the high-resolution images captured by
a monocular front facing camera are depth maps, seman-
tic segmentation masks, optical ﬂow data, and surface nor-
mals. These components are provided at a resolution of
720 pixels, with depth maps covering a range from 0
1280
to 100 m. Additionally, the dataset incorporates exact drone
pose, velocity, and acceleration data for each frame.

⇥

The DDOS dataset is systematically divided into train-
ing, validation, and testing subsets, consisting of 300, 20,
and 20 ﬂights, respectively. It features pixel-wise segmen-
tation masks for ten distinct classes, enabling in-depth anal-
ysis of various obstacles and environmental elements. Fig-

7330Image

Depth

Segmentation

Flow

Surface normals

Figure 1. Examples from our DDOS dataset. This ﬁgure showcases an overview of the DDOS dataset’s multifaceted annotations. It
includes RGB images from drone ﬂights, depth maps (0–100 m), pixel-wise semantic segmentation, optical ﬂow and surface normals,
illustrating the dataset’s richness and diversity.

ure 1 displays select examples from the dataset, demonstrat-
ing the diversity of classes represented. More examples are
available in Appendix B. The methodological approach to
dataset generation and the classiﬁcation scheme are further
elaborated in Section 4, providing insight into the dataset’s
design choices and structure.

4. Data Generation

DDOS is generated using AirSim [32], an open-source
drone simulator. DDOS is composed of two environments
that mimic real-world scenarios. The ﬁrst environment re-
sembles a small suburban town, featuring dense trees and
numerous power lines, replicating the challenges faced dur-
ing drone ﬂights in residential areas. The second environ-
ment represents a park setting, incorporating elements such
as a football ﬁeld with ﬂoodlights, a beach volleyball court,
dense trees as well as ofﬁce buildings. These environments
collectively offer diverse obstacles and structures, allowing
researchers to develop and evaluate algorithms capable of
addressing the complexities associated with different real-
world environments. By encompassing characteristics like
dense tree coverage, power lines, and varying weather con-
ditions, the dataset provides a comprehensive platform for

advancing obstacle segmentation and depth estimation al-
gorithms for safe and effective drone ﬂights.

Flight trajectories To construct each ﬂight trajectory, a
random starting location (x0, y0, z0), within the environ-
ment bounds is selected. Subsequently, multiple interme-
diate target points (xt, yt, zt) are generated within prede-
ﬁned relative bounding boxes, dictating the areas to which
the drone navigates. Flight characteristics, are varied across
different ﬂights, providing diversity in the dataset. During
each ﬂight, observations are recorded at a rate of 10 Hz for
a duration of 10 seconds. These observations encompass a
rich set of data, including images, depth maps, pixel-wise
object segmentation, optical ﬂow, and surface-normals.

In order to promote relatively safe
Collision avoidance
ﬂight paths, we developed a dynamic obstacle detection al-
gorithm to modify intermediate targets in response to po-
tential collision risks. This algorithm utilizes the most re-
cent ground truth depth map obtained during the recorded
ﬂight observations. By empirically determining a thresh-
old, objects that are deemed too close trigger updates to the
intermediate targets. The updated targets are strategically
adjusted based on the detected obstacle’s location, causing
the drone to navigate away from the identiﬁed collision risk.

7331Figure 2. Distribution of class labels within DDOS. DDOS effectively captures the presence of various thin object classes, which are
characterized by a relatively sparse distribution of pixels within each image. Despite their limited pixel coverage, these thin object classes
are well-represented in DDOS, ensuring comprehensive coverage and enabling robust training and evaluation of algorithms speciﬁcally
designed to address the challenges posed by such objects.

This obstacle avoidance approach is not ﬂawless, especially
when dealing with thin structures, occasional collisions re-
sulting in crashes still occur. In such cases, the observations
associated with the crash event are discarded and the ﬂight
process is restarted to ensure data integrity. It is important
to note, the collision avoidance mechanism is purposefully
designed to be lax, as near misses and even minor crashes
can offer valuable data points for training purposes.

Post-processing To uphold the overall integrity of the
dataset and exclude instances of undesired behavior, addi-
tional validation criteria are applied after ﬂight generation.
These criteria serve to ﬁlter out scenarios where the drone
becomes stuck or encounters unusual situations, such as be-
coming entangled in trees. By incorporating these post-
ﬂight validation steps, the dataset ensures that the collected
observations reﬂect reliable and meaningful ﬂight behav-
iors, enabling robust algorithm training, and evaluation.

Data augmentation We do not augment the dataset with
additional transformations or modiﬁcations, such as chro-
matic aberration, added lens ﬂares, corruption, or noise,
during the data collection process. The decision to exclude
these augmentation techniques at the initial phase ensures
that the dataset remains in its original state, preserving the
inherent characteristics and properties of the collected data.
Instead, we provide the ﬂexibility to incorporate these aug-
mentation techniques at a later stage, if deemed necessary,
during algorithm development and evaluation.

Weather DDOS encompasses diverse environmental and
weather conditions, including sunny, dusk, and brightly lit
night scenes, along with rain, fog, snow, and changes due
to wet surfaces and snow cover. These conditions challenge
vision-based algorithms with reduced visibility and altered
surface characteristics, such as increased reﬂectivity from
snow and glare from wet roads, complicating object detec-
Including these varied scenarios
tion and scene analysis.

is essential for developing models that adapt and perform
consistently in all real-world settings.

Classes Objects are systematically classiﬁed based on
their signiﬁcance for drone navigation. Ultra Thin encom-
passes wires and cables; Thin Structures includes poles and
signs; Small Mesh pertains to fences and nets; and Large
Mesh covers objects such as transmission towers that per-
mit drone passage. Additionally, Trees, Buildings, Vehicles,
and Animals are categorized based on straightforward char-
acteristics. The Other class encompasses diverse objects
like bus stops, post boxes, chairs, and tables. Background
refers to elements such as the ground and sky, providing
context within the scene.

5. Dataset Statistics

In this section, we provide a comprehensive analysis of key
properties inherent in the DDOS dataset. Figure 2 illustrates
the distribution of annotations across diverse classes within
DDOS. Signiﬁcantly, the dataset adeptly captures and rep-
resents various classes of thin structures, even when these
objects occupy a relatively small number of pixels in each
image. This nuanced representation ensures that DDOS of-
fers a substantial and well-balanced dataset for thin object
classes. This richness in diversity is paramount for facili-
tating thorough analysis, robust algorithm training, and ef-
fective evaluation, particularly in addressing the challenges
associated with thin structures in real-world scenarios. The
carefully crafted distribution of classes within DDOS con-
tributes to its utility as a reliable benchmark for advancing
the capabilities of algorithms designed for thin structure de-
tection and segmentation.

In our continued investigation, we analyze the pitch and
roll angles observed during ﬂight sessions. As depicted in
Figure 3, there is a wide range of pitch and roll angles,
indicating signiﬁcant variations in the drone’s orientation
across the dataset. Despite the drone’s primary forward mo-

7332Figure 3. Distribution of pitch and roll angles. The colors rep-
resent the intensity levels, with warmer colors indicating higher
occurrences. Flight characteristics vary between each ﬂight, as
highlighted by the diverse pitch and roll degrees. The pitch is neg-
ative when the drone is accelerating forward and positive when
braking or to go backwards. Emergency braking is often accom-
panied with a sharp turn, either to the left or to the right.

tion, the angles demonstrate a notable diversity. This vari-
ety in orientation provides valuable perspectives for evaluat-
ing algorithms under different ﬂight conditions. The broad
distribution of pitch and roll angles emphasizes the DDOS
dataset’s ability to mimic real-world ﬂying scenarios, where
drones encounter various orientations. This characteristic
enhances the dataset’s utility for training and evaluating al-
gorithms to ensure consistent performance amidst the ori-
entation challenges that drones face in actual ﬂights.

To gain an intuitive understanding of the spatial distri-
bution of ﬂight paths within an environment, we visually
present a subset of the recorded trajectories in Figure 4.
The depicted ﬂight paths showcase a diverse array of pat-
terns, ranging from sharp turns and straight lines to curved
trajectories. These variations authentically capture the com-
plexity and dynamic nature of the simulated environments.
Furthermore, an overhead view of the relative ﬂight paths,
presented in Figure 5, offers a normalized perspective with
a common starting point and direction. This visualization
emphasizes the diverse ﬂight trajectories and patterns ob-
served across individual ﬂights, providing a comprehen-
sive overview of the spatial dynamics inherent in DDOS.
Such a representation is instrumental in offering insights
into the intricate navigation challenges that algorithms must
address, reinforcing the dataset’s efﬁcacy in training and
evaluating models under diverse and realistic conditions.

Expanding our analysis, we explore the distributions of
altitude and speed during the ﬂights, along with the distri-

Figure 4. Illustrated ﬂight paths. The ﬁgure presents a collection
of 50 randomly selected ﬂight paths conducted within the same
environment. The paths exhibit signiﬁcant variations in trajectory,
highlighting the diverse nature of drone ﬂights.

bution of depth recorded in the depth maps, as illustrated
collectively in Figure 6. Examining the altitude distribution
reveals that the drone operates at varying heights, encom-
passing low-level ﬂights near the ground to higher altitudes.
The distribution of speed elucidates a spectrum of velocities
encountered during the ﬂights, showcasing diverse ﬂight be-
haviors and maneuvering speeds. Moreover, the depth dis-
tribution offers insights into the range and distribution of
depth values recorded in the depth maps, shedding light on
the variations in perceived depth across the dataset.

6. Depth Metrics

We propose a novel set of depth metrics speciﬁcally tailored
for drone applications, namely the absolute relative depth
estimation error for each distinct class. To illustrate, we in-
troduce the absolute relative depth error metric for the Ultra
Thin class within the DDOS dataset. This metric quanti-
ﬁes the accuracy of depth estimation speciﬁcally for objects
classiﬁed as Ultra Thin in the DDOS dataset.

AbsRelultra thin =

1
Nultra thin

ˆdi

di  
di

Nultra thin

i=1  
 
X
 
 
 

 
 
 
 
 

(1)

Here, AbsRelultra thin represents the absolute relative
depth estimation error for the Ultra Thin class. Nultra thin
denotes the total number of samples (pixels) in the Ultra
Thin class, while di and ˆdi represent the ground truth depth
and estimated depth for the i-th pixel sample, respectively.
The formula calculates the average absolute relative differ-
ence between the ground truth and estimated depths for all
samples in the Ultra Thin class. Trivially, extending this ap-

7333(a) Distribution of ﬂight altitude.

(b) Distribution of ﬂight speed.

(c) Distribution of depth.

Figure 6. Distributions of altitude, speed, and depth. The distri-
butions show variation across ﬂights. Depth over 100 m is ignored.

This suite is tailored to assess the performance of our meth-
ods at a ﬁner class level, offering a more detailed under-
standing of their capabilities.

We utilize three different baselines, BinsFormer [19],
SimIPU [18] and DepthFormer [20]. BinsFormer proposes
a novel framework for monocular depth estimation by for-
mulating it as a classiﬁcation-regression task, employing
a transformer [37] decoder to generate adaptive bins [5].
SimIPU introduces a pre-training strategy for spatial-aware
visual representation, utilizing point clouds for improved
spatial information in contrastive learning. DepthFormer
addresses supervised monocular depth estimation by lever-
aging a transformer for global context modeling, incorpo-
rating an additional convolution branch, and introducing a
hierarchical aggregation module.

When evaluated using standard depth metrics, the base-
lines exhibit satisfactory performance, as shown in Table 2.
However, using our class-speciﬁc depth metrics, shown in
Table 3 and depicted in Figure 7, unveils substantial chal-
lenges in achieving accurate depth estimations for certain
object classes. Speciﬁcally, the Ultra Thin category is ex-
ceptionally challenging, with all tested methods failing to
provide accurate depth estimations.

These ﬁndings highlight the importance of develop-
ing methodologies that are speciﬁcally tailored to enhance
depth estimation accuracy for ultra-thin structures, partic-
ularly in drone-based applications. Future research should
focus on addressing these challenges, aiming to enhance the
precision and reliability of depth estimations for these chal-
lenging scenarios.

Figure 5. Overhead view of relative ﬂight paths with a nor-
malized starting point. In this visualization the starting location
and direction have been normalized to highlight the various rel-
ative shapes of the ﬂight paths. The actual starting locations are
randomly initialized, as shown in Figure 4.

proach to all classes, the general formula for class-speciﬁc
depth metrics becomes:

AbsRelclass =

1
Nclass

ˆdi

di  
di

Nclass

i=1  
 
X
 
 
 

 
 
 
 
 

(2)

Assessing class-speciﬁc absolute relative depth errors re-
veals how well depth estimation algorithms perform, espe-
cially for intricate structures like wires and cables. This
method offers a detailed evaluation, highlighting how algo-
rithms manage the challenges unique to various structures
seen from drone viewpoints. The motivation for this nu-
anced approach stems from the recognition that traditional
metrics fail to adequately represent difﬁcult-to-detect ob-
stacles, such as wires, due to their low pixel count. A thor-
ough investigation into these aspects is essential to accu-
rately gauge the efﬁcacy and robustness of vision systems.

7. Baselines

We use a set of commonly-used depth metrics to evaluate
the effectiveness of the baselines. These metrics include
fundamental measures such as accuracy under the thresh-
old ( i < 1.25i, i = 1, 2, 3), which assesses the model’s
performance within proximity thresholds. Additionally, we
use mean absolute relative error (AbsRel), mean squared
relative error (SqRel), root mean squared error (RMSE),
root mean squared log error (RMSElog), mean log10 error
(log10) and scale-invariant logarithmic error (SILog).

Moreover, in pursuit of a more nuanced evaluation, we
leverage our newly proposed suite of metrics known as
mean absolute relative class error metrics (AbsRelclass).

7334Model

BinsFormer [19]
SimIPU [18]
DepthFormer [20]

 1 "
0.632
0.760
0.860

 2 "
0.792
0.918
0.958

 3 "
0.845
0.964
0.981

AbsRel

#

0.265
0.225
0.136

RMSE

#
16.211
7.095
5.831

log10

#
0.139
0.070
0.050

RMSElog

#

0.466
0.245
0.190

SILog

#
38.009
22.715
18.101

SqRel

#
6.387
3.302
1.614

Table 2. Monocular depth estimation performance. The table compares BinsFormer, SimIPU, and DepthFormer across various tra-
ditional performance metrics. Notably, DepthFormer outperforms the other baselines across all metrics, showcasing seemingly great
performance in accurately estimating depth. The arrows indicate desired outcome.

Model

BinsFormer [19]
SimIPU [18]
DepthFormer [20]

Ultra
Thin

0.945
1.036
0.998

Thin
Structures

0.216
0.317
0.229

Small
Mesh

0.129
0.178
0.115

Large
Mesh

0.209
0.233
0.177

Trees Buildings Vehicles Animals Other Background

0.248
0.380
0.206

0.137
0.198
0.121

0.141
0.204
0.120

0.150
0.176
0.121

0.141
0.184
0.128

0.257
0.122
0.082

Table 3. Class-wise absolute relative depth errors. Each baseline’s performance is evaluated per class, with lower values indicating
better performance. DepthFormer achieves the lowest errors for the larger classes but completely fails to estimate depth for Ultra Thin. All
methods severely struggle for the Ultra Thin class.

Input Image

Ground Truth

BinsFormer [19]

SimIPU [18]

DepthFormer [20]

Figure 7. Depth estimation performance of baselines. This qualitative assessment underscores the challenges faced by state-of-the-art
methods in accurately estimating depth, particularly for the Ultra Thin class. The results showcases the shared difﬁculty encountered by
all methods in capturing the Ultra Thin class. This emphasizes the intricate nature of accurately discerning depth for such instances.

8. Conclusion

In summary, we introduce the DDOS dataset along with
novel drone-speciﬁc depth metrics, marking a pivotal ad-
vancement in the ﬁeld of autonomous drone navigation. The
DDOS dataset addresses the critical challenges of detecting
thin structures and operating under varied weather condi-
tions, thereby ﬁlling an essential gap in the current scope of
drone research. Through a detailed analysis of the dataset
and the deployment of tailored evaluation metrics, we pro-
vide a nuanced methodology for systematically assessing
the performance of depth estimation algorithms in drone-
speciﬁc scenarios.

These efforts establish a new standard for future inves-
tigations aimed at enhancing the safety and efﬁciency of
drone navigation through superior depth estimation and se-
mantic segmentation techniques. The introduction of the
DDOS dataset and corresponding metrics not only propels
forward the development of drone technology but also ex-
tends the potential for computer vision applications within
aerial environments. Our work lays a crucial groundwork
for future innovations, steering the creation of algorithms
that adeptly navigate the complexities of real-world set-
tings, thus amplifying the functional prowess of drones
across a multitude of industries.

7335References

[1] Rabab Abdelfattah, Xiaofeng Wang, and Song Wang. Ttpla:
An aerial-image dataset for detection and segmentation of
transmission towers and power lines. In Proceedings of the
Asian Conference on Computer Vision, 2020. 2, 3

[2] Stuart M Adams and Carol J Friedland. A survey of un-
manned aerial vehicle (uav) usage for imagery collection
In 9th international
in disaster research and management.
workshop on remote sensing for disaster response, pages 1–
8, 2011. 1

[3] Babankumar Bansod, Rangoli Singh, Ritula Thakur, and
Gaurav Singhal. A comparision between satellite based and
drone based remote sensing technology to achieve sustain-
able development: A review. Journal of Agriculture and En-
vironment for International Development (JAEID), 111(2):
383–407, 2017. 1

[4] Taha Benarbia and Kyandoghere Kyamakya. A literature re-
view of drone-based package delivery logistics systems and
their implementation feasibility. Sustainability, 14(1):360,
2021. 1

[5] Shariq Farooq Bhat, Ibraheem Alhashim, and Peter Wonka.
Adabins: Depth estimation using adaptive bins. In Proceed-
ings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, pages 4009–4018, 2021. 7

[6] Holger Caesar, Varun Bankiti, Alex H. Lang, Sourabh Vora,
Venice Erin Liong, Qiang Xu, Anush Krishnan, Yu Pan, Gi-
ancarlo Baldan, and Oscar Beijbom. nuscenes: A multi-
modal dataset for autonomous driving. In CVPR, 2020. 2
[7] Joshua Candamo, Rangachar Kasturi, Dmitry Goldgof, and
Sudeep Sarkar. Detection of Thin Lines using Low-Quality
Video from Low-Altitude Aircraft in Urban Settings. IEEE
Transactions on Aerospace and Electronic Systems, 45(3):
937–949, 2009. 2

[8] Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo
Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe
Franke, Stefan Roth, and Bernt Schiele. The cityscapes
dataset for semantic urban scene understanding. In Proceed-
ings of the IEEE Conference on Computer Vision and Pattern
Recognition (CVPR), 2016. 2

[9] Sharifah Mastura Syed Mohd Daud, Mohd Yusmiaidil Put-
era Mohd Yusof, Chong Chin Heo, Lay See Khoo, Man-
sharan Kaur Chainchel Singh, Mohd Shah Mahmood, and
Hapizah Nawawi. Applications of drone in disaster manage-
ment: A scoping review. Science & Justice, 62(1):30–42,
2022. 1

[10] Milan Erdelj, Enrico Natalizio, Kaushik R. Chowdhury, and
Ian F. Akyildiz. Help from the Sky: Leveraging UAVs for
IEEE Pervasive Computing, 16(1):
Disaster Management.
24–32, 2017.

[11] Mario Arturo Ruiz Estrada and Abrahim Ndoma. The uses of
unmanned aerial vehicles–uav’s-(or drones) in social logis-
tic: Natural disasters response and humanitarian relief aid.
Procedia Computer Science, 149:375–383, 2019. 1

[12] Michael Fonder and Marc Van Droogenbroeck. Mid-air: A
multi-modal dataset for extremely low altitude drone ﬂights.
In Conference on Computer Vision and Pattern Recognition
Workshop (CVPRW), 2019. 2, 3

[13] Vipul Garg, Suman Niranjan, Victor Prybutok, Terrance
Pohlen, and David Gligor. Drones in last-mile delivery: A
systematic review on efﬁciency, accessibility, and sustain-
ability. Transportation Research Part D: Transport and En-
vironment, 123:103831, 2023. 1

[14] Timnit Gebru, Jamie Morgenstern, Briana Vecchione, Jen-
nifer Wortman Vaughan, Hanna Wallach, Hal Daum´e Iii, and
Kate Crawford. Datasheets for datasets. Communications of
the ACM, 64(12):86–92, 2021. 11

[15] Andreas Geiger, Philip Lenz, and Raquel Urtasun. Are we
ready for autonomous driving? the kitti vision benchmark
suite. In 2012 IEEE conference on computer vision and pat-
tern recognition, pages 3354–3361. IEEE, 2012. 2

[16] Yoshio Inoue. Satellite-and drone-based remote sensing of
crops and soils for smart farming–a review. Soil Science and
Plant Nutrition, 66(6):798–810, 2020. 1

[17] James R. Kellner, John Armston, Markus Birrer, K. C. Cush-
man, Laura Duncanson, Christoph Eck, Christoph Falleger,
Benedikt Imbach, Kamil Kr´al, Martin Krˇuˇcek, Jan Trochta,
Tom´aˇs Vrˇska, and Carlo Zgraggen. New opportunities for
forest remote sensing through ultra-high-density drone lidar.
Surveys in Geophysics, 40:959–977, 2019. 1

[18] Zhenyu Li, Zehui Chen, Ang Li, Liangji Fang, Qinhong
Jiang, Xianming Liu, Junjun Jiang, Bolei Zhou, and Hang
Zhao. Simipu: Simple 2d image and 3d point cloud un-
supervised pre-training for spatial-aware visual representa-
tions. In Proceedings of the AAAI Conference on Artiﬁcial
Intelligence, pages 1500–1508, 2022. 7, 8

[19] Zhenyu Li, Xuyang Wang, Xianming Liu, and Junjun Jiang.
Binsformer: Revisiting adaptive bins for monocular depth
estimation. arXiv preprint arXiv:2204.00987, 2022. 7, 8
[20] Zhenyu Li, Zehui Chen, Xianming Liu, and Junjun Jiang.
Depthformer: Exploiting long-range correlation and local in-
formation for accurate monocular depth estimation. Machine
Intelligence Research, pages 1–18, 2023. 7, 8

[21] Ye Lyu, George Vosselman, Gui-Song Xia, Alper Yilmaz,
and Michael Ying Yang. Uavid: A semantic segmentation
dataset for uav imagery. ISPRS Journal of Photogrammetry
and Remote Sensing, 165:108–119, 2020. 2, 3

[22] Ratnesh Madaan, Daniel Maturana, and Sebastian Scherer.
Wire detection using synthetic data and dilated convolutional
networks for unmanned aerial vehicles. In 2017 IEEE/RSJ
International Conference on Intelligent Robots and Systems
(IROS), pages 3487–3494. IEEE, 2017. 2, 3

[23] Alina Marcu, Vlad Licaret, Dragos Costea, and Marius
Leordeanu. Semantics through time: Semi-supervised seg-
mentation of aerial videos with iterative label propagation.
In Proceedings of the Asian Conference on Computer Vision,
2020. 2, 3

[24] Moritz Menze and Andreas Geiger. Object scene ﬂow for au-
tonomous vehicles. In Proceedings of the IEEE conference
on computer vision and pattern recognition, pages 3061–
3070, 2015. 2

[25] Payal Mittal, Raman Singh, and Akashdeep Sharma. Deep
learning-based object detection in low-altitude uav datasets:
A survey. Image and Vision computing, 104:104046, 2020.
1

7336In 2020 IEEE/RSJ International Conference
visual slam.
on Intelligent Robots and Systems (IROS), pages 4909–4916.
IEEE, 2020. 2, 3

[39] World

Economic

discriminatory

Forum Global
Future
Coun-
cil on Human Rights 2016–2018.
How to pre-
learning.
vent
https://www.weforum.org/whitepapers/how-
to - prevent - discriminatory - outcomes - in -
machine-learning, 2018. 11

in machine

outcomes

[26] Norzailawati Mohd Noor, Alias Abdullah, and Mazlan
Hashim. Remote sensing uav/drones and its applications for
urban areas: A review. In IOP conference series: Earth and
environmental science, page 012003. IOP Publishing, 2018.
1

[27] Ishan Nigam, Chen Huang, and Deva Ramanan. Ensem-
ble knowledge transfer for semantic segmentation. In 2018
IEEE Winter Conference on Applications of Computer Vision
(WACV), pages 1499–1508. IEEE, 2018. 2, 3

[28] Yalong Pi, Nipun D Nath, and Amir H Behzadan. Convolu-
tional neural networks for object detection in aerial imagery
for disaster response and recovery. Advanced Engineering
Informatics, 43:101009, 2020. 1

[29] Chengyi Qu, Francesco Betti Sorbelli, Rounak Singh, Prasad
Environmentally-aware and
Calyam, and Sajal K Das.
energy-efﬁcient multi-drone coordination and networking
IEEE Transactions on Network and
for disaster response.
Service Management, 2023. 1

[30] Giulia Rizzoli, Francesco Barbato, Matteo Caligiuri, and
Pietro Zanuttigh. Syndrone-multi-modal uav dataset for ur-
In Proceedings of the IEEE/CVF Interna-
ban scenarios.
tional Conference on Computer Vision, pages 2210–2220,
2023. 2, 3

[31] Aamina Shah, Komali Kantamaneni, Shirish Ravan, and
Luiza C Campos. A systematic review investigating the use
of earth observation for the assistance of water, sanitation
and hygiene in disaster response and recovery. Sustainabil-
ity, 15(4):3290, 2023. 1

[32] Shital Shah, Debadeepta Dey, Chris Lovett, and Ashish
Kapoor. Airsim: High-ﬁdelity visual and physical simula-
tion for autonomous vehicles. In Field and Service Robotics,
2017. 4, 12

[33] Adam Stambler, Gary Sherwin, and Patrick Rowe. Detec-
tion and Reconstruction of Wires Using Cameras for Air-
In 2019 International Conference
craft Safety Systems.
on Robotics and Automation (ICRA), pages 697–703, 2019.
ISSN: 1050-4729. 2

[34] Pei Sun, Henrik Kretzschmar, Xerxes Dotiwalla, Aurelien
Chouard, Vijaysai Patnaik, Paul Tsui, James Guo, Yin Zhou,
Yuning Chai, Benjamin Caine, et al. Scalability in perception
for autonomous driving: Waymo open dataset. In Proceed-
ings of the IEEE/CVF conference on computer vision and
pattern recognition, pages 2446–2454, 2020. 2

[35] Lina Tang and Guofan Shao. Drone remote sensing for
Journal of Forestry Re-

forestry research and practices.
search, 26(4):791–797, 2015. 1

[36] Ashley Varghese, Jayavardhana Gubbi, Hrishikesh Sharma,
and P Balamuralidhar. Power infrastructure monitoring and
In 2017
damage detection using drone captured images.
international joint conference on neural networks (IJCNN),
pages 1681–1687. IEEE, 2017. 2, 3

[37] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszko-
reit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia
Polosukhin. Attention is all you need. Advances in neural
information processing systems, 30, 2017. 7

[38] Wenshan Wang, Delong Zhu, Xiangwei Wang, Yaoyu Hu,
Yuheng Qiu, Chen Wang, Yafei Hu, Ashish Kapoor, and Se-
bastian Scherer. Tartanair: A dataset to push the limits of

7337