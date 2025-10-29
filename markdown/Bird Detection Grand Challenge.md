ICASSP 2023 Signal Processing Grand Challenges

Received 12 December 2023; accepted 4 February 2024. Date of publication 19 March 2024;
date of current version 18 June 2024. The review of this article was arranged by Associate Editor H. Vicky Zhao.

Digital Object Identiﬁer 10.1109/OJSP.2024.3379073

The Drone-vs-Bird Detection Grand Challenge
at ICASSP 2023: A Review of Methods and
Results

ANGELO COLUCCIA

1 (Senior Member, IEEE), ALESSIO FASCISTA 1 (Member, IEEE), LARS SOMMER 2,

ARNE SCHUMANN2, ANASTASIOS DIMOU 3, AND DIMITRIOS ZARPALAS 3
1Department of Innovation Engineering, University of Salento, 73100 Lecce, Italy
2Fraunhofer Center for Machine Learning, Fraunhofer IOSB, 76131 Karlsruhe, Germany
3Centre for Research and Technology Hellas, Visual Computing Lab, Information Technologies Institute, 57001 Thessaloniki, Greece

CORRESPONDING AUTHOR: ANGELO COLUCCIA (e-mail: angelo.coluccia@unisalento.it)

ABSTRACT This paper presents the 6th edition of the “Drone-vs-Bird” detection challenge, jointly or-
ganized with the WOSDETC workshop within the IEEE International Conference on Acoustics, Speech
and Signal Processing (ICASSP) 2023. The main objective of the challenge is to advance the current
state-of-the-art in detecting the presence of one or more Unmanned Aerial Vehicles (UAVs) in real video
scenes, while facing challenging conditions such as moving cameras, disturbing environmental factors, and
the presence of birds ﬂying in the foreground. For this purpose, a video dataset was provided for training the
proposed solutions, and a separate test dataset was released a few days before the challenge deadline to assess
their performance. The dataset has continually expanded over consecutive installments of the Drone-vs-Bird
challenge and remains openly available to the research community, for non-commercial purposes. The
challenge attracted novel signal processing solutions, mainly based on deep learning algorithms. The paper
illustrates the results achieved by the teams that successfully participated in the 2023 challenge, offering
a concise overview of the state-of-the-art in the ﬁeld of drone detection using video signal processing.
Additionally, the paper provides valuable insights into potential directions for future research, building upon
the main pros and limitations of the solutions presented by the participating teams.

INDEX TERMS Deep learning, drone detection, image and video signal processing, unmanned aerial vehi-
cles (UAV).

I. INTRODUCTION
Unmanned Aerial Vehicles (UAVs), commonly known as
drones, have gained immense popularity and found diverse
applications in recent years, encompassing areas such as mon-
itoring, environmental protection, support to communication
systems [1], [2], [3]. While their versatility and capabilities
offer numerous beneﬁts, there is a growing need for effective
UAV detection systems due to the concerns raised on vari-
ous aspects, including security, safety, and privacy [4]. The
2023 annual report of the Federal Aviation Administration
(FAA) of US reported multiple incidents over the recent years
[5], mainly caused by malicious or suspicious usage, or in-
advertent misuse of UAVs, proving the severity and timely
importance of the problem.

Detecting and identifying unauthorized drones can help
preventing potential threats, safeguarding critical infrastruc-
tures, and protecting individual privacy. The increasing uti-
lization of UAVs highlights the need not only for implement-
ing and regulating rules regarding drone ﬂights, but also for
establishing effective UAV detection systems to accurately
localize intruders or unauthorized drones. For instance, certain
sensitive areas, such as airports, military bases, government
facilities, or nuclear power plants, require heightened secu-
rity measures. Similarly, critical infrastructure such as power
plants, oil reﬁneries, and telecommunications networks are
crucial to the functioning of societies [6].

UAV detection systems play a crucial role also in ensuring
air trafﬁc safety as the risk of collisions with manned aircrafts

766

© 2024 The Authors. This work is licensed under a Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 License. For more information, see
https://creativecommons.org/licenses/by-nc-nd/4.0/

VOLUME 5, 2024

also rises with the ever increasing number of UAVs in the
airspace [7]. By identifying and tracking UAVs, authorities
can implement appropriate measures to prevent collisions,
maintain the integrity of ﬂight paths, and reduce the potential
for accidents or disruptions.

UAVs have been misused also for various illegal activities,
including smuggling drugs, contraband, or weapons across
borders [8]. In this respect, UAV detection systems can aid
law enforcement agencies in identifying and apprehending
individuals involved in such activities. Rapid detection and re-
sponse enhance the effectiveness of border control operations,
mitigating the risks associated with illegal trafﬁcking.

The increasing need for high performing counter-drone sys-
tems stresses the need for innovative approaches providing
reliable automatic detection and identiﬁcation of drones, in a
variety of environments and scenarios. In terms of operational
capacity, an effective detection system must be able to de-
tect threats from diverse drone types, including custom-made
and entirely new designs. Another challenge faced by current
systems is the adaptation to a new or evolving environment
(i.e. weather, sunlight, vegetation, surrounding infrastructures,
presence of birds) which can make systems ineffective. Mod-
ern Counter-UAV systems build upon a number of detection
technologies (e.g. visual, RF, radar, acoustic) [9], [10], to
overcome speciﬁc challenges and limitations of each individ-
ual modality.

While there is a lot of ongoing research on the subject
from many different communities, e.g. [11], [12], [13], [14],
[15], [16], the availability of datasets that can be exploited for
training UAV detection systems is rather limited. Some or-
ganizations and companies may have proprietary ones. These
datasets may include real-world scenarios, proprietary detec-
tion algorithms, and sensor data. However, access to such
datasets is typically restricted and may require partnerships
or agreements with the data owners.

In-house data collection provides the advantage of tailoring
the dataset to speciﬁc needs and capturing data in speciﬁc op-
erational contexts, but it is an expensive and time-consuming
task in terms of equipment, regulations, privacy preserva-
tion, and most importantly annotation. As a matter of fact,
capturing data utilizing diverse UAV types, under different
environmental conditions, including birds that may interfere
with the detection capabilities, is not possible for most re-
search teams.

In this paper, we present

Collaboration and data-sharing initiatives among indus-
try stakeholders, government agencies, and research com-
munities can help increase data availability for UAV de-
tection system training.
the
Drone-vs-Bird Detection Grand Challenge, an initiative that
combined data capturing campaigns from European projects,
in order to offer to the research community a compre-
hensive dataset on visual capturings of UAVs, manually
annotated, aiming to promote research on the domain. The
Challenge focuses on providing the means to advance the
drone detection state of the art (SoA), by seeking for inno-
vative signal processing solutions for video data sequences.

Since the launching of Drone-vs-Bird Detection Grand Chal-
lenge, numerous academic groups or companies have received
the dataset, in order to train or evaluate their own drone
detection methods.

The ﬁrst edition of the International Workshop on Small-
Drone Surveillance, Detection and Counteraction Techniques
(WOSDETC) [17] was organized in 2017 as part of IEEE In-
ternational Conference on Advanced Video and Signal based
Surveillance (AVSS), held in Lecce, Italy. In conjunction with
the workshop, the grand challenge called Drone-vs-Bird De-
tection Challenge was launched. In 2019, a second edition
of the challenge was organized, again as part of WOSDETC
and co-located with the 16th edition of AVSS held in Taipei,
Taiwan [18]. A third edition of the Drone-vs-Bird challenge
was organized in 2020, initially planned as part of the 17th
edition of AVSS in Washington DC, USA, but then run as
virtual event due to the COVID-19 pandemic [19]. The fourth
edition of the challenge was organized in conjunction with the
17th AVSS in 2021 as a virtual event [20]. The ﬁfth edition
of the challenge was held in conjunction with ICIAP 2021
(May 2022) [21] in Lecce. The present work extends the short
(two-page) paper in [22] and provides a more deep overview
of the methodologies and outcomes of the 6th edition of the
Grand Challenge, which was held as part of ICASSP 2023 on
Rhodes Island, Greece.

A very high number of access requests to the dataset have
been ﬁled since the challenge inception, by numerous teams
from various countries. This reﬂects the global reach and pop-
ularity of the challenge, showcasing the widespread interest
and involvement of research communities from around the
world. However, it is worth noting that only a handful of teams
manage to submit valid results, conﬁrming the general difﬁ-
culty of the detection task and the need of further research and
advancements in the topic. To offer a quantitative summary, in
the 2023 edition, we received approximately 100 requests for
the dataset, had about 20 registrations for the grand challenge,
and received 8 successful submissions by the deadline from 4
distinct teams.

II. THE DRONE-VS-BIRD DETECTION
CHALLENGE DATASET
In this section, we give a comprehensive description of the
dataset utilized for the Drone vs. Bird Detection Grand Chal-
lenge at ICASSP 2023. We begin by outlining its primary
attributes, including the diverse range of video sequences
encompassed, the various models of drones and disturbing
objects encountered, as well as the variability observed within
the considered scenes. Furthermore, we provide a concise
summary of other drone datasets freely accessible in existing
literature to offer a holistic perspective. Lastly, we present a
brief overview of the participation history, spanning from the
challenge’s ﬁrst edition to the latest 2023 edition.

A. THE CHALLENGE TRAINING DATASET
The Drone-vs-Bird Detection Challenge dataset encompasses
a diverse collection of 77 video sequences, serving as training

VOLUME 5, 2024

767

COLUCCIA ET AL.: DRONE-VS-BIRD DETECTION GRAND CHALLENGE AT ICASSP 2023: A REVIEW OF METHODS AND RESULTS

FIGURE 2. Examples of drone types present in the training set, i.e., Parrot
Disco, 2 custom ﬁxed-wing drones, DJI Inspire, DJI Phantom, DJI Mavic, DJI
Matrice and 3DR Solo Robotics.

FIGURE 1. General operational scenario of the Drone-vs-Bird Detection
Challenge.

data for all participating teams. This dataset has undergone
progressive evolution over the editions of the challenge.
Initially, a portion of the videos was obtained through experi-
mental campaigns conducted within the SafeShore project,1
utilizing MPEG4-coded static cameras. These recordings
were subsequently augmented by additional sequences con-
tributed by the Fraunhofer IOSB research institute, sourced
from various locations across Germany. In 2020, the AL-
ADDIN project2 introduced 45 more videos, incorporating
the use of moving cameras for acquisition. Overall, the train-
ing dataset to date comprises a combination of sequences
captured with both static and moving cameras, featuring di-
verse resolutions ranging from 720 × 576 to 3840 × 2160
pixels. Note that static cameras allow a straightforward ap-
plication of motion detection methods as initial detector or
in addition to appearance based detection methods, while
sequences recorded by moving cameras require a camera
motion compensation.

Each sequence contains an average of approximately 1,384
frames, with an average of 1.12 annotated drones per frame.
As illustrated in Fig. 2, the dataset encompasses eight dis-
tinct types of commercial drones, including Parrot Disco, DJI
Inspire, DJI Phantom, DJI Mavic, DJI Matrice, 3DR Solo
Robotics, and two custom ﬁxed-wing drones. Among these,
three types possess ﬁxed wings, while the remaining ﬁve ex-
hibit rotary wings.

The training dataset is comprised of sequences provided by
different research institutes, which recorded their data at dif-
ferent locations under varying conditions, thus offering a large
variety of scenes and backgrounds. It features the presence of

1The project “SafeShore” has been granted funding from the European
Union’s Horizon 2020 research and innovation programme, with grant agree-
ment No. 700643.

2The project “ALADDIN” has been granted funding from the European
Union’s Horizon 2020 research and innovation programme, with grant agree-
ment No. 740859.

FIGURE 3. Sample frames extracted from the training videos showing the
large variability of the dataset.

both static and moving camera sequences, of different lengths,
with frame characteristics changing also within a same se-
quence (e.g., the camera may ﬁrst point to the sky but then
follow the drone on the land, with trees background or mar-
itime scene or others). More speciﬁcally, scenes include urban
areas, woodlands, agricultural areas, urban areas and rivers
in Central Europe, maritime areas as well as Mediterranean
landscapes and cities, resulting in varying levels of difﬁculty
for the detection algorithms. A diverse range of backgrounds
is observed, including sky, buildings, water surfaces, and
different kinds of vegetation, i.e., trees, grassland, bushes,
and rocks. The dataset further incorporates different weather
conditions such as cloudy and sunny, and different recording
times such as daytime, dawn and nighttime. Moreover, it en-
compasses challenges such as direct sun glare and variations
in camera characteristics, as depicted in Fig. 3. While drones
are annotated in the dataset, birds, often appearing as main
disturbing objects, speciﬁcally in more than one-third of the
sequences, are not annotated (further discussion on this point
will be provided in Section VI).

The distance between the drones and the camera exhibits
signiﬁcant variability across and within the videos, leading
to considerable variations in drone sizes, as showcased in

768

VOLUME 5, 2024

FIGURE 4. Distribution of drone sizes across the ground truth annotations
in the training dataset.

Fig. 4. The drone sizes range from as small as 15 pix-
els to over 1,000,000 pixels. The majority of annotated
drones have sizes less than 162 pixels or fall within the
range of 162 to 322 pixels. The presence of small-sized
drones poses a particularly challenging detection task. To
facilitate the training process, each video sequence is accom-
panied by a separate annotation ﬁle, available on GitHub
(at https://github.com/wosdetc/challenge). This ﬁle contains
information on the frames in which drones enter the scenes,
along with their precise locations expressed as bounding
w h]. In this notation, (topx,
boxes in the form of [ topx topy
topy) represents the coordinates of the top right corner, while
w and h indicate the width and height of the bounding box,
respectively. While drones are annotated in the dataset, birds,
often appearing as main disturbing objects, i.e. in more than
one third of the sequences, are not annotated.

B. THE CHALLENGE TEST DATASET
The challenge test set encompasses an additional 30 video
sequences, for which no annotations are provided. Among
these, 16 video sequences are inherited from initial editions
of the challenge. Most of the locations depicted in these se-
quences are also present in the training set and exhibit similar
characteristics. To increase the difﬁculty, the test set has been
enriched with new video sequences that introduce novel back-
grounds and two distinct types of rotary drones. Furthermore,
the test set features the presence of additional disturbing ob-
jects, such as planes, and includes scenarios where drones are
located against structured backgrounds, as shown in Fig. 5.
To ensure a fair evaluation, sequences exceeding 30 seconds
in duration have been shortened to prevent a few individual
videos from dominating the whole evaluation process.

As concerns the level of overlap between the training and
test sets in the Drone-vs-Bird dataset, it is worth highlighting
that most sequences in the test set are recorded at completely
unseen locations, whereas the remaining sequences that share
similar (but not identical) backgrounds with the training set
have been acquired using different perspectives and recording
times. The latter is indeed a common strategy to minimize the

FIGURE 5. Sample frames extracted from the test videos showcase
notable differences compared to the training set.

overlap in the datasets while balancing the efforts necessary
to conduct the acquisition campaign (in some practical cases,
in fact, changing to completely different scenes may be not
even possible due to constraints on the movement of hardware
equipment). Thus, the Drone-vs-Bird dataset allows to assess
the effectiveness of the proposed detection methods (includ-
ing, for instance, aspects such as the generalization ability)
with nearly-zero overlap between training and test sets.

The dataset, including both the training set and the test
set, is openly available for download. However, to access the
dataset, interested individuals must ﬁrst sign a Data Usage
Agreement (DUA) to comply with the terms and conditions
regarding the use and handling of the data. For convenience,
the annotations for the dataset can be accessed at the following
URL: https://github.com/wosdetc/challenge.

C. OTHER DRONE DATASETS
For the sake of completeness, we now review other publicly
available drone detection datasets, in comparison with the
Drone-vs-Bird dataset. It is important to note that datasets
based on other sensor modalities are not considered in this
overview, although some of the datasets may also include
EO (Electro-Optical) or IR (Infrared) imagery in addition to
(visible-light) video sequences.

The ﬁrst dataset we discuss is the Drone Dataset: Ama-
teur Unmanned Air Vehicle Detection, released in 2019 [23].
This dataset includes over 4000 images featuring DJI Phan-
tom drones. Images have a resolution between 300 × 168
pixels and 4 k, and the dataset also comprises images with
non-drone objects.

The Small Target Detection database (USC-GRAD-STDdb)
[24] was built using 115 video segments downloaded from
YouTube. The frames have a resolution of 1280 × 720 pixels,
with speciﬁc annotations available for about 25, 000 frames.
They include more than 56, 000 small objects, categorized
as drones, birds, boats, vehicles, and people. Out of the 115
video segments, 57 contain either drones or birds, while the
Drone-vs-Bird dataset speciﬁcally considers the simultaneous
presence of both drones and birds in the scene.

VOLUME 5, 2024

769

COLUCCIA ET AL.: DRONE-VS-BIRD DETECTION GRAND CHALLENGE AT ICASSP 2023: A REVIEW OF METHODS AND RESULTS

The Purdue UAV dataset [25] is a smaller dataset com-
prising only ﬁve video sequences, for a total of 1829 frames.
These video sequences were recorded using a custom airframe
with a camera and have a frame rate of 30 frames per sec-
ond. Images have a resolution that is either 1920 × 1080 or
1280 × 960 pixels. Moreover, the annotations for the ground
truth are openly available for download.

Another dataset worth mentioning is the Flying Object
Detection from a Single Moving Camera dataset [26]. The
dataset consists of 20 video sequences, with each image
752 × 480 pixels and containing,
having a resolution of
on average, two similar objects that challenge the detection
task. The video sequences were acquired with a commercial
UAV mounting a standard camera, resulting in varying
drone appearances caused by changing orientations, lighting
conditions, and other
this dataset
includes 20 video sequences featuring aircraft sourced
from YouTube, exhibiting image resolutions ranging from
640 × 480 to 1280 × 720 pixels.

factors. Furthermore,

A more recent dataset is the Real World Object Detection
Dataset for Quadcopter Unmanned Aerial Vehicle Detection
[27]. This dataset encompasses an extensive collection
of 51 446 training images and an additional 5375 images
speciﬁcally allocated for
testing purposes. The images
themselves were procured through a combination of internet
downloads and author-captured content, all adjusted to adhere
640 × 480 pixels. Within the
to a uniform resolution of
training set, about 52, 676 different instances of drones can
be found. Conversely, the test set is composed of about
2863 drone instances, alongside 2750 images void of any
drone presence. To expedite the annotation procedure, an
innovative semi-automated labeling pipeline was effectively
implemented. Notably, within the training set, approximately
40.8% of the drones are conﬁned to dimensions smaller than
32 × 32 pixels, while about 23.4% exceed the threshold of
96 × 96 pixels. In the test set, similar proportions reveal
that about 36.3% of the drones are of smaller dimensions
than 32 × 32 pixels, while a noteworthy 28.3% surpass the
dimension of 96 × 96 pixels.

The Anti-UAV Challenge dataset [28] was released in 2020
as part of the IEEE Conference on Computer Vision and
Pattern Recognition (CVPR). Unlike the Drone vs. Bird De-
tection Challenge, the Anti-UAV Challenge focuses on the
task of tracking a single object. The dataset consists of a total
of 160 video sequences, including IR and EO imagery. The IR
images have a resolution of 640 × 512 pixels, while the EO
images have a resolution of 1920 × 1080 pixels. About 100
video sequences have been annotated, serving as training data
for tracking algorithms. The videos were recorded through
a rotating platform equipped with a static camera. Conse-
quently, the acquisition campaign is limited only to a few
selected scenarios. Additionally, this dataset focuses on four
speciﬁc drone types: DJI Inspire, DJI Phantom, DJI MavicAir,
and DJI MavicPRO.

reconstructing 3D ﬂight trajectories, using as acquisition sys-
tem an ad-hoc network of cameras. These datasets consist of
ﬁve separate datasets. The ﬁrst four datasets accounts for the
presence of an hexacopter captured with different cameras,
while the ﬁfth dataset involves three different types of drones.
The datasets consider different acquisition setups, with a num-
ber of cameras changing from 4 to 7. Moreover, the ﬂight
duration varies between 2 and 10 minutes. In terms of anno-
tations, they are provided in the form of a single point for the
ﬁrst four datasets. Compared to the Drone vs. Bird Detection
Challenge dataset, these Multi-view drone tracking datasets
are smaller in size and lack the inclusion of diverse envi-
ronmental settings. Not least, they are tailored for different
goals, such as tracking of single objects or reconstruction of
3D trajectories.

The VISIODECT dataset [30], released in 2022, comprises
20,924 sample images and associated annotations, encom-
passing six drone models operating in three distinct scenarios
(cloudy, sunny, and evening), at various altitudes and dis-
tances ranging from 30 m to 100 m. The data is available
in three different ﬁle formats (txt, xml, csv) and was gener-
ated over 1 a and 8 months at 12 different locations. Each
video sequence was converted into JPEG image frames with
dimensions of 852 × 480 pixels. These frames were orga-
nized and stored in repositories, each representing a speciﬁc
model class and scenario sub-class. To enhance data quality,
a teams of professionals cleaned each repository by manu-
ally selecting image frames that did not feature drones in
the background. Data annotation was conducted by manually
delineating bounding boxes around each image ﬁle, result-
ing in the creation of corresponding label ﬁles. To maintain
a consistent naming convention and minimize errors, label
ﬁles for each scenario sub-class were named to align with
their respective image ﬁles and stored in repositories ac-
cordingly. Differently from the Drone-vs-Bird dataset, the
VISIODECT does not include sequences acquired from mov-
ing cameras and does not account for the presence of birds or
similar objects.

Another very recent dataset is the Anti-UAV Detection and
Tracking from Dalian University of Technology (DUT) [13].
The whole dataset divides in two separated subsets: one for
detection and the other for tracking. The detection dataset,
which has a similar scope as the Drone-vs-Bird, accounts for
10,000 images in total, in which the training, testing, and
validation sets have 5200, 2200 and 2600 images, respec-
tively. All frames and images have been manually annotated.
Image resolution spans from 160 × 240 to 3744 × 5616,
offering a large variability in the UAV sizes across different
sequences. There are more than 35 different UAV models
appearing in the detection dataset, ﬂying in outdoor environ-
ments including sky, dark clouds, jungles, high-rise buildings,
residential buildings, farmland, and playgrounds. Compared
to our Drone-vs-Bird dataset, the DUT dataset mainly lacks
the presence of birds or other disturbing ﬂying objects.

On the other hand,

the Multi-view drone tracking
datasets [29] were proposed to deal with the problem of

The Halmstad dataset represents another valuable source
of video sequences meant for UAV detection [31]. Data have

770

VOLUME 5, 2024

TABLE 1. Attributes Covered by the Overviewed UAV Datasets

been captured at three airports in Sweden (Halmstad, Gothen-
burg, and Malmö) and comprise 650 video sequences, includ-
ing also some non-copyrighted material from the YouTube
channel “Virtual Airﬁeld operated by SK678387” used to en-
rich the target categories (mainly airplanes and helicopters).
The dataset features only 3 different types of UAVs and all
the videos have a resolution of 640 × 512 pixels, and a total
duration of 10 seconds each. The maximum distance at which
UAVs are captured from the ﬁxed cameras is about 200 m.
Given the limited scenario considered for the construction of
the dataset (airports only), it does not provide high variability
in terms of scenes and weather conditions.

The USC Drone Dataset represents another freely-avilable
dataset speciﬁcally constructed for video-based object detec-
tion and tracking [32]. It contains only 30 sequences, all
recorded at the USC campus. The sequences include the
presence of a single drone model but span a variety of differ-
ent backgrounds, different angles of acquisition and variable
weather conditions. The dataset has the objective of capturing
real UAV attributes such as fast maneuvering, occlusions, and
high illumination, just to mention a few. All video sequences
have a ﬁxed resolution of 1920 × 1080, with each individual
video lasting approximately one minute. To partially compen-
sate for the limited variability in terms of scenes and drone
models, the dataset also uses model-based data augmentation
techniques that synthesize training images and annotate loca-
tion of each drone within frames automatically.

In Table 1 we compare the major attributes of our dataset
against different datasets identiﬁed in the literature. From ex-
perience in the ﬁeld, a number of challenges can be identiﬁed,
which need to be overcome for detecting drones “in the wild”.
The latter, in fact, are found i) of differing types, shapes,
sizes and models (varying from tiny to large ones), ii) within
a variety of backgrounds that may cause false positives, iii)
within varying environmental, weather and time (day, dawn,

sunshine, cloudy, dark) conditions, iv) using stable footage
or not, v) in videos of differing resolution and drone sizes
in pixels, vi) with the simultaneous presence of other ﬂying
objects (e.g., birds) that might cause false positives. Table 1
shows that indeed our proposed dataset is the one that meets
all the expected criteria.

III. DESCRIPTION OF TASKS AND EVALUATION METRICS
A. DETECTION TASK
The detection task of the Drone-vs-Bird Detection Challenge
2023 requires that participating teams submit a set of result
ﬁles. These ﬁles should encompass each video sequence, with
explicit indications of the frame numbers in which drones
were detected. Alongside the frame numbers, the predicted
position of the drones within the frame must be provided
in the same format of annotations, namely bounding boxes
w h]. Additionally, result ﬁles
denoted by [ topx
should include conﬁdence scores for each frame, aiding in the
assessment of the algorithm’s uncertainty on its predictions.
In cases where a frame does not contain any reported detec-
tions, it will be assumed that no drones were detected in that
particular frame.

topy

While the use of additional training data is permitted, teams
must provide detailed descriptions regarding the quantity and
nature of the supplementary data employed. It is essential for
teams relying on additional data to submit an additional result
of their method, indicating the performance achieved solely
using the provided training data. However, the ultimate evalu-
ation and ranking for the challenge will be based on the overall
best score achieved, irrespective of the utilization of additional
data. The ultimate goal of the algorithms should be to achieve
precise and accurate localization of drones, ensuring that the
estimated bounding boxes closely correspond to the actual
UAVs positions.

VOLUME 5, 2024

771

COLUCCIA ET AL.: DRONE-VS-BIRD DETECTION GRAND CHALLENGE AT ICASSP 2023: A REVIEW OF METHODS AND RESULTS

YOLOv5m6 [33] was employed as drone detection model.
The model was trained on four different drone detection
datasets to increase the model’s generalization ability. In ad-
dition, a new synthetic drone detection dataset, which consists
of random background images and randomly placed drone
objects, was employed to improve the detection performance
in case of complex and unseen backgrounds. The model was
trained for 10 epochs using the default YOLOv5 training con-
ﬁguration, while the image scale was set to 1344 pixels. While
an image based drone classiﬁcation model was utilized in their
previous work to improve the detection accuracy, OBSS AI
modiﬁed this classiﬁer to classify image sequences [34]. For
this purpose, an object tracker generated tracks for detected
objects. Then, eight instances of a track were fed to a sequence
classiﬁer model, computing a drone probability. This drone
probability was combined with the object detectors’ drone
probability using geometric mean. To train the classiﬁer, a
dataset was semi-automatically created by using their detector
and tracker to generate object tracks from training video se-
quences. These tracks were exported and manually labeled as
drone, bird, and other. To overcome missing detections in case
of complex backgrounds, OBSS utilized a template match-
ing approach. Therefore, historical data were stored for each
tracked object. If the object detector failed, a template match-
ing algorithm was applied near the last object location in a
small search region, i.e. image width / 10 × image height / 10.
Predicted bounding boxes were then fed to the sequence clas-
siﬁer model to calculate the drone probability of the object.

IIT (Indian Institute of Technology Jammu) proposed a de-
tection scheme comprised of three stages. Initially, YOLOv7
[35] is applied as drone detection model, which was trained
on 60 videos from the Drone-vs-Bird challenge train set. To
reduce the number of false positive detections, detections
were ﬁltered based on the conﬁdence score in the subsequent
stage. For this purpose, IIT estimated the number of drones
n for each sequence and only considered the corresponding
n bounding boxes with the highest conﬁdence scores. The
number of drones per sequence was derived from the number
of detections for each image throughout the entire sequence.
In the last stage, a CSRT tracker [36] was employed to account
for missed detections in complex environments. As the tracker
is less reliable than YOLOv7 in detecting accurate bounding
boxes, IIT proposed a scheme to fuse bounding boxes esti-
mated by both YOLOv7 and the tracker. If YOLOv7 did not
detect a drone, detections from the previous frame were used
to initialize trackers. Then bounding boxes predicted by the
tracker were used until detections become available again. To
identify false positive detections, the IoU between detections
and the tracker’s bounding boxes was used. If the IoU was
less than 0.3, detections were considered as false positive
detections. Additional details can be found in the ICASSP
paper [37].

DU (Dongguk University, South Korea) adapted the
medium-size YOLOv8 [38] model for drone detection. To
account for various drone sizes, a multi-scale image fusion
(MSIF) [39] approach was employed. MSIF extracts features

FIGURE 6. Intersection over Union (IoU) metric that captures the goodness
in predicting a drone position in a frame.

B. EVALUATION METRICS
The evaluation process for the Drone-vs-Bird Detection Chal-
lenge employs the widely-adopted Average Precision (AP)
metric, which is commonly utilized in object detection tasks
such as the COCO object detection challenge. The AP metric
is based on the Intersection over Union (IoU) criterion, which
measures the overlap between the estimated bounding box
and the ground truth bounding box surrounding the UAV in
the scene. The IoU is calculated as the ratio of the area of
overlap between the two boxes to the total area of their union,
as shown Fig. 6.

To determine the accuracy of detections, a threshold (typi-
cally 0.5) is applied to the IoU. If the IoU between a detected
UAV and a ground truth annotation exceeds the threshold, it
is considered a true positive detection. Conversely, detections
with an IoU below the threshold are counted as false posi-
tives. Any ground truth annotations that are not assigned to a
detection are regarded as false negatives, representing missed
detections. By calculating the area under the precision-recall
curve, the AP metric provides a comprehensive evaluation of a
detector’s performance, capturing the trade-off between preci-
sion and recall. This single metric thus effectively summarizes
the overall precision-recall characteristics of each proposed
algorithm. It is important to note that the test sequences are
made available to participants one week prior to the submis-
sion deadline; teams are requested to run their algorithms on
these test data and submit the results. Eventually, teams that
realize the performance of their algorithms are qualitatively
inadequate, typically withdraw from the challenge and refrain
from submitting any results.

IV. DRONE DETECTION ALGORITHMS
In the following the drone detection algorithms used for sub-
missions are brieﬂy discussed.

OBSS AI (OBSS Teknoloji Ankara, Turkey) proposed a
drone detection framework that comprises an initial deep
learning based drone detector, a sequence classiﬁer and
template matching. Based on their previous approach [21],

772

VOLUME 5, 2024

TABLE 2. Results of Drone-vs-Bird Detection Challenge 2023

TABLE 3. Detailed Comparison for Each Team in the Drone-vs-Bird
Detection Challenge 2023

for three different scales of the input image, which are fused
into one feature map through bottom-up and top-down struc-
tures. The combined feature map was then used as input of the
YOLOv8 model. To improve the detection accuracy in case
of small drones, the P2 layer of the backbone was added to
the feature pyramid of YOLOv8 due to strong spatial local
features. DU further applied data augmentation to increase
the number of drone appearances in the training images. For
this, a copy and paste scheme [40] was employed. Cropped
and scaled drones were randomly located under the condi-
tion that the new location did not overlap with an already
drone-occupied area. For training, every ﬁfth frame of the
Drone-vs-Bird challenge train set was used and the image
scale was set to 640 pixels. The YOLOv8 model was trained
for 93 epochs with a batch size of 16. For inference, the
image scale was set to 1280 pixels. Furthermore, DU applied
horizontal ﬂipping and multi-scale augmentation. For more
details, we refer the reader to the ICASSP paper [41].

Note that all approaches are based on detection methods ap-
plied on single images. Hence, no approach considers motion
based detection methods to identify possible drone locations.
However, OBSS employs temporal information by using an
additional tracker, which can be useful in case of distant
drones or complex backgrounds.

SNU (Shandong Normal University, China) adapted Single
Shot Multi-Box Detector (SSD) [42] as detection model. To
account for small drones, SNU added a shallow feature pyra-
mid network and attention module. For training, SNU used
images from the Drone-vs-Bird challenge train set and set the
image scale to 300 pixels. The interested reader is referred to
the ICASSP paper [43] for more details.

TABLE 4. Number of Detections and Recall

V. PERFORMANCE ASSESSMENT
A. ANALYSIS OF THE RESULTS
The ﬁnal ranking of the Drone-vs-Bird challenge 2023 is
reported in Table 2, showing the AP for each submission. The
winning entry was submitted by OBSS AI, which substan-
tially outperformed the other participating teams.

The AP values for each sequence are given in Ta-
ble 3. For
this, we only considered the best submis-
sion from each team. OBSS achieved the best AP on
all sequences, exhibiting good detection results on most
sequences. However, poor AP values are obtained in

case of scenes with weak contrast between drone and
structured background, e.g. VID_20210606_143947_04 and
VID_20210606_141511_01. The detection results for IIT
clearly differ for the different sequences. While high detection
accuracies are achieved on several sequences, all drones are
missed in other sequences. DU and SNU exhibit poor AP
values on most sequences, while good detection results are
only achieved for scenes with large UAVs and simple (non-
structured) background.

The number of submitted detections and overall recall are
given in Table 4. The test sequences comprise about 18000

VOLUME 5, 2024

773

COLUCCIA ET AL.: DRONE-VS-BIRD DETECTION GRAND CHALLENGE AT ICASSP 2023: A REVIEW OF METHODS AND RESULTS

TABLE 5. Detailed Comparison for Each Team in the Drone-vs-Bird
Detection Challenge 2023

TABLE 6. AP for Different Drone Sizes

TABLE 7. Recall for Different Drone Sizes

considers motion-based detection methods to identify possible
drone locations. However, OBSS partially exploits temporal
information by using an additional tracker, which can be use-
ful in case of distant drones or complex backgrounds.

To further analyze the detection results, we computed the
AP values and recall rates for different drone sizes (see Table 6
and Table 7, respectively). OBSS achieved the best AP values
and highest recall rates for all drone sizes. Though the recall
rate and AP increases with larger drones, OBSS shows a high
recall rate and good AP even for small drones whose size is
less than 162 pixels. The APs and recall rates obtained by
IIT are in the same range for different UAV sizes, yielding the
best results for drone sizes in the range between 162 and 322
pixels. While the recall rates for DU are similar for different
drone sizes, the AP values are worse for smaller drones. This
indicates that more false positive detections are caused in case
of small drone sizes. The results for SNU show that only large
drones are detected, whereas all small drones are missed. One
reason for this is the used image scale, which results in clearly
down-scaled input images, so that small drones only comprise
a few pixels.

A detailed analysis of the occurring errors is given in Fig.
7. A series of precision recall curves (PRCs) is given for
each team. C75, C50 and Loc are the PRCs for IoU thresh-
olds of 0.75, 0.5 and 0.1, respectively. Due to the less strict
IoU threshold, Loc depicts inaccurately localized detections.
BG points out false positive detections caused by the back-
ground, while FN shows remaining false negative detections.
For OBSS, the remaining errors are caused by inaccurate lo-
calization, false positive detections due to background clutter
and missed detections, while no error source clearly domi-
nates. The main error source for IIT is missed detections. One
reason for this could be the applied ﬁltering scheme, as several
drone detections might be ﬁltered out. In addition to the high
number of missed detections, DU exhibits numerous false
positive detections caused by the background. This indicates
that the applied model is not able to accurately distinguish
between drones and clutter objects. For SNU, the errors are

VOLUME 5, 2024

annotated drones. While the number of detections submitted
by OBSS and DU exceed the number of annotations, IIT and
SNU submitted clearly less detections. Thus, OBSS exhibits
a high recall rate, whereas IIT and SNU show poor recall
rates. Though the high number of submitted detections, DU
achieved a poor recall rate, which indicates that most detec-
tions are false positive detections.

The recall rate for each sequence are listed in Table 5.
OBSS exhibits good recall rates except for some sequences,
which comprise scenes with weak contrast between drone and
background. The recall rates for IIT clearly differ for the dif-
ferent sequences. For some sequences, all drones are correctly
detected, while all drones are missed in other sequences. DU
and in particular SNU show numerous scenes without any or
only few detections.

Notice that static cameras allow a straightforward appli-
cation of motion detection methods as initial detector or in
addition to appearance-based detection methods, while se-
quences recorded by moving cameras require a camera motion
compensation. Moreover, all approaches are based on detec-
tion methods applied on single images; hence, none of them

774

FIGURE 7. Error analysis using the COCO evaluation toolbox [44]. For each team, a series of precision recall curves (PRCs) is given. C75 is the PRC for an
IoU of 0.75 to accept detections as true positives, while C50 is the PRC for an IoU of 0.5 as used within this challenge. Loc depicts localization errors. For
this, the IoU criterion is set to 0.1. BG shows false positive detections caused by the background and FN illustrates the remaining false negatives.

mainly due to false negative detections. As already discussed
one reason for this is the inappropriate down-scaling of the
input images. The high numbers of missed detections for IIT,
DU and SNU indicate the poor generalization ability of the
applied models. In contrast to OBSS, these teams considered
only the Drone-vs-Bird challenge train set and no additional
datasets for training. However, the test set comprises multiple
scenes with partially complex backgrounds, which are unseen
during training and thus, may cause missed detections due to
unexpected appearances of drones. This indicates that most
approaches are robust only in case of sequences comparable
with those in the training set, e.g. drones with the sky as
background, while performance may signiﬁcantly change in
case of variations in the UAV sizes and complexity of the
background, as structured background often yields weak con-
trast to drones. Considering that the best performance have
been obtained by using additional datasets for training, it is
apparent that the diversity in the training data plays an impor-
tant role for both adaptability and generalizability.

Examples of qualitative detection results for all teams are
given in Figs. 8 and 9. Note that only detections with a
conﬁdence score above 0.5 are considered. In case of large
drones and unstructured background, all approaches achieved
good detection results (see Fig. 8). However, small drones as

well as drones in front of background are only detected by
OBSS and IIT or only by OBSS. In case of more complex
backgrounds or weak contrast between drone and background,
all approaches have issues to correctly detect drones. Besides
more diverse training data or novel data augmentation tech-
niques, the usage of temporal information could be beneﬁcial
for such scenarios.

B. DISCUSSION
The 6th edition of the Drone-vs-Bird challenge involved the
participation of four distinct research teams. The analyses and
results reported in Section V-A clearly demonstrated that the
algorithm proposed by the OBSS team signiﬁcantly outper-
formed the approaches proposed by the other participating
teams over all the sequences provided in the test set. From
a more technical point of view, the superiority of the de-
tection framework proposed by OBSS can be ascribed to its
ability to mitigate effects introduced by mobile cameras and
to detect distant drones. Although all the teams applied only
appearance-based detectors on single frames without consid-
ering more extended motion information, OBSS inserted a
tracking approach in the processing loop that helped the pro-
posed method to identify with high probability the presence

VOLUME 5, 2024

775

COLUCCIA ET AL.: DRONE-VS-BIRD DETECTION GRAND CHALLENGE AT ICASSP 2023: A REVIEW OF METHODS AND RESULTS

FIGURE 8. Qualitative examples for OBSS AI (green), IIT (red), Dongguk (blue) and Shandong (yellow) showing good detection results in case of large
drones and unstructured background.

FIGURE 9. Qualitative examples for OBSS AI (green), IIT (red), Dongguk (blue) and Shandong (yellow) showing reasons for missed detections.

776

VOLUME 5, 2024

of drones even in case of small and blurry appearance. More-
over, the drone detection model used by OBSS was trained
on four different drone detection datasets to increase the
model generalization ability and was further augmented with
a synthetic drone dataset that lead to improved the detection
performance in case of complex and unseen backgrounds. On
the other hand, the methods proposed by IIT, DU and SNU
teams could be considered effective only for sequences with
simple (non-structured) backgrounds and with large drones
appearing at same instant. When facing scenes with more
complex backgrounds or smaller drones, all the methods from
IIT, DU, and SNU tend to exhibit a too high number of missed
detections, while the method proposed by OBSS is able to
limit the number of miss detections or false alarms, though at
the price of a reduced AP.

Overall, for the case of large drones and unstructured
background, all approaches achieved satisfactory detection
results. However, small drones as well as drones in front
of backgrounds are detected by OBSS and, only in part,
by IIT. All approaches suffered in case of more complex
backgrounds or weak contrast between drone and back-
ground. One of the primary difﬁculty arises from managing
mobile cameras and detecting distant drones. Another im-
portant aspect to highlight is that most of the algorithms
do not explicitly incorporate birds in a supervised manner
during the design phase due to the absence of annotated
bird data. Consequently, instances where multiple birds are
present in test sequences (including scenes with entire ﬂocks)
tend to result in increased false alarms across all methods,
as birds share small visual characteristics with small (dis-
tant) UAVs. Not least, the majority of the models adopted
by the participating teams were trained on a few different
real and synthetic datasets, thus exhibiting a rather poor
generalization capability.

VI. CONCLUSION
This paper presented an overview of the outcomes from the
6th edition of the Drone vs Birds Detection Grand Chal-
lenge at ICASSP 2023. The four methods proposed by the
participating teams exhibit distinct design elements, lead-
ing to a complementary set of interesting aspects. Notably,
the primary difﬁculties arise from managing mobile cam-
eras and detecting distant drones. Another important aspect
to highlight is that most of the algorithms do not explicitly
incorporate birds in a supervised manner during the design
phase due to the absence of annotated bird data. Consequently,
instances where multiple birds are present in test sequences
(including scenes with entire ﬂocks) tend to result in increased
false alarms across all methods, as birds share small visual
characteristics with small (distant) UAVs. Incorporating bird
targets into the training dataset has been proved a challenging
and labour-intensive task. In drone tracking footage, birds
appear as small and blurry ﬂying objects, often not easy to
be identiﬁed in single images as bird without utilising motion
information within a sequence. Furthermore, in case of ﬂock
of birds, annotation would be a very time-consuming task, that

VOLUME 5, 2024

cannot promise a satisfactory accuracy. Addressing this issue
necessitates devising strategies to integrate bird data, partic-
ularly given the visual similarity between distant ﬁxed-wing
UAVs and birds. It should be also kept in mind that the goal
is drone detection, not classiﬁcation of the rest of the scene.
Designing a method for more general object detection and
classiﬁcation would lead to a different approach for future
extension of the challenge, incorporating an additional class
representing birds at the design stage, as well as other classes
for similar ﬂying objects (e.g., airplanes). To this aim, a ﬁrst
step could be to annotate birds only in videos where their
appearance is evident enough both for the sake of annota-
tion and useful training of the detection method, which also
makes it possible to consider the adoption of semi-automatic
annotation tools. Additionally, videos solely of birds (avail-
able on the Internet) could be used to train a method with
bird appearance features. Another possibility would be to
generate videos with drones and birds, by augmenting a
drone video with synthetically generated ﬂying bird(s). This
would alleviate the hassle of bird annotation, but requires to
construct suitable methods for realistic bird ﬂights genera-
tion. All such aspects warrant further exploration and will
be a focal point in upcoming editions of the Drone vs. Bird
Detection Challenge.

More generally, understanding the main factors that con-
tribute to the evident performance variations exhibited by each
algorithm across different sequences is an important direc-
tion of further research, in particular for what concerns the
ability to cope with arbitrarily-complex backgrounds. Future
editions of the challenge could also incorporate additional as-
sessments: besides the mentioned multi-class extension, other
performance aspects such as computational efﬁciency (includ-
ing real-time capabilities) could be investigated. The use of a
shared Docker container installed on a remote machine (e.g.,
using one of the cloud facilities) could be a viable solution
to compare the runtime of the proposed algorithms on the
same hardware platform and assess whether they are suit-
able for real-time implementation. The latter is expected to
evolve into a crucial requirement in the future editions of the
challenge, given the increasing importance of promptly de-
tecting drones as a fundamental prerequisite for modern UAV
detection systems.

In conclusion, all the inquiries above aim to unravel the
intricate trade-offs inherent in the multitude of approaches
and methodological combinations adopted for drone detection
based on video signal processing, contributing to a deeper un-
derstanding of their underlying mechanisms and highlighting,
at the same time, the most promising research directions.

REFERENCES
[1] H. V. Nguyen, H. Rezatoﬁghi, B.-N. Vo, and D. C. Ranasinghe, “On-
line UAV path planning for joint detection and tracking of multiple
radio-tagged objects,” IEEE Trans. Signal Process., vol. 67, no. 20,
pp. 5365–5379, Oct. 2019.

[2] K. K. Nguyen, A. Masaracchia, V. Sharma, H. V. Poor, and
T. Q. Duong, “RIS-Assisted UAV communications for IoT with wireless
power transfer using deep reinforcement learning,” IEEE J. Sel. Topics
Signal Process., vol. 16, no. 5, pp. 1086–1096, Aug. 2022.

777

COLUCCIA ET AL.: DRONE-VS-BIRD DETECTION GRAND CHALLENGE AT ICASSP 2023: A REVIEW OF METHODS AND RESULTS

[3] A. Fascista, “Toward integrated large-scale environmental monitoring
using WSN/UAV/Crowdsensing: A review of applications, signal pro-
cessing, and future perspectives,” Sensors, vol. 22, no. 5, 2022, Art.
no. 1824. [Online]. Available: https://www.mdpi.com/1424-8220/22/5/
1824

[4] V. Chamola, P. Kotesh, A. Agarwal, N. N. Gupta, and M. Guizani, “A
comprehensive review of unmanned aerial vehicle attacks and neutral-
ization techniques,” Ad Hoc Netw., vol. 111, 2021, Art. no. 102324.
[Online]. Available: https://www.sciencedirect.com/science/article/pii/
S1570870520306788

[5] “UAS sightings report,” 2023. [Online]. Available: https://www.faa.

gov/uas/resources/public_records/uas_sightings_report

[6] “A drone tried to disrupt the power grid. It won’t be the last,” 2021.
[Online]. Available: https://www.wired.com/story/drone-attack-power-
substation-threat/

[7] “Drone operator above Tesla Giga Berlin spoils routine descent for
passenger plane,” 2022. [Online]. Available: https://www.teslarati.com/
tesla-giga-berlin-drone-operator-berlin-brandenburg-airport-plane/
[8] “Security forces foil Narco-Terrorism bid,” 2022. [Online]. Available:

https://tinyurl.com/5hy2h6v7

[9] I. Guvenc, F. Koohifar, S. Singh, M. L. Sichitiu, and D. Matolak, “De-
tection, tracking, and interdiction for amateur drones,” IEEE Commun.
Mag., vol. 56, no. 4, pp. 75–81, Apr. 2018.

[10] A. Coluccia, G. Parisi, and A. Fascista, “Detection and classiﬁcation
of multirotor drones in radar sensor networks: A review,” Sensors, vol.
20, no. 15, 2020, Art. no. 4172. [Online]. Available: https://www.mdpi.
com/1424-8220/20/15/4172

[11] A. G. Haddad, M. A. Humais, N. Werghi, and A. Shoufan, “Long-range
visual UAV detection and tracking system with threat level assessment,”
in Proc. 46th Annu. Conf. IEEE Ind. Electron. Soc., 2020, pp. 638–643.
[12] B. K. S. Isaac-Medina, M. Poyser, D. Organisciak, C. G. Willcocks, T. P.
Breckon, and H. P. H. Shum, “Unmanned aerial vehicle visual detection
and tracking using deep neural networks: A performance bench-
mark,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. Workshops, 2021,
pp. 1223–1232.

[13] J. Zhao, J. Zhang, D. Li, and D. Wang, “Vision-based Anti-UAV de-
tection and tracking,” IEEE Trans. Intell. Transp. Syst., vol. 23, no. 12,
pp. 25323–25334, Dec. 2022.

[14] J. Li, D. H. Ye, M. Kolsch, J. P. Wachs, and C. A. Bouman, “Fast and
robust UAV to UAV detection and tracking from video,” IEEE Trans.
Emerg. Topics Comput., vol. 10, no. 3, pp. 1519–1531, Jul.–Sep. 2022.
[15] S. Samaras et al., “Deep learning on multi sensor data for counter
UAV applications–A systematic review,” Sensors, vol. 19, no. 22, 2019,
Art. no. 4837. [Online]. Available: https://www.mdpi.com/1424-8220/
19/22/4837

[16] T. Müller et al., “Drone detection, recognition, and assistance system
for counter-UAV with VIS, radar, and radio sensors,” Proc. SPIE, vol.
12096, pp. 94–108, 2022.

[17] A. Coluccia et al., “Drone-vs-bird detection challenge at

IEEE
AVSS2017,” in Proc. 14th IEEE Int. Conf. Adv. Video Signal Based
Surveill., 2017, pp. 1–6.

[18] A. Coluccia et al., “Drone-vs-bird detection challenge at IEEE AVSS
2019,” in Proc. 16th IEEE Int. Conf. Adv. Video Signal Based Surveill.,
2019, pp. 1–7.

[19] A. Coluccia et al., “Drone vs. bird detection: Deep learning algorithms
and results from a grand challenge,” Sensors, vol. 21, no. 8, 2021, Art.
no. 2824.

[20] A. Coluccia et al., “Drone-vs-bird detection challenge at

IEEE
AVSS2021,” in Proc. 17th IEEE Int. Conf. Adv. Video Signal Based
Surveill., 2021, pp. 1–8.

[21] A. Coluccia et al., “Drone-vs-bird detection challenge at ICIAP 2021,”

in Proc. Conf. Image Anal. Process. Workshops, 2022, pp. 410–421.

[22] A. Coluccia et al., “Drone-vs-bird detection grand challenge at
ICASSP2023,” in Proc. IEEE Int. Conf. Acoust., Speech, Signal Pro-
cess., 2023, pp. 1–2.

[23] M. C. Aksoy, A. S. Orak, H. M. özkan, and B. Selimoglu, “Drone
dataset: Amateur unmanned air vehicle detection,” Mendeley Data, V4,
2019, doi: 10.17632/zcsj2g2m4c.4.

[24] B. Bosquet, M. Mucientes, and V. Brea, “STDNet: A ConvNet for small
target detection,” in Proc. 29th Brit. Mach. Vis. Conf., 2018, Art. no.
253.

[25] J. Li, D. H. Ye, T. Chung, M. Kolsch, J. Wachs, and C. Bouman,
“Multi-target detection and tracking from a single camera in unmanned
aerial vehicles (UAVs),” in Proc. IEEE/RSJ Int. Conf. Intell. Robots
Syst., 2016, pp. 4992–4997.

[26] A. Rozantsev, V. Lepetit, and P. Fua, “Flying objects detection from
a single moving camera,” in Proc. IEEE Conf. Comput. Vis. Pattern
Recognit., 2015, pp. 4128–4136.

[27] M. Ł. Pawełczyk and M. Wojtyra, “Real world object detection dataset
for quadcopter unmanned aerial vehicle detection,” IEEE Access, vol.
8, pp. 174394–174409, 2020.

[28] “Anti-UAV challenge.” 2024. [Online]. Available: https://anti-uav.

github.io/

[29] J. Li, J. Murray, D. Ismaili, K. Schindler, and C. Albl, “Reconstruc-
tion of 3D ﬂight trajectories from ad-hoc camera networks,” in Proc.
IEEE/RSJ Int. Conf. Intell. Robots Syst., 2020, pp. 1621–1628.

[30] S. O. Ajakwe, V. U. Ihekoronye, D.-S. Kim, and J. M. Lee, “DRONet:
Multi-tasking framework for real-time industrial facility aerial surveil-
lance and safety,” Drones, vol. 6, no. 2, 2022, Art. no. 46.

[31] F. Svanström, C. Englund, and F. Alonso-Fernandez, “Real-time drone
detection and tracking with visible, thermal and acoustic sensors,” in
Proc. 25th Int. Conf. Pattern Recognit., 2021, pp. 7265–7272.

[32] Y. Chen, P. Aggarwal, J. Choi, and C.-C. J. Kuo, “A deep learning ap-
proach to drone monitoring,” in Proc. Asia-Paciﬁc Signal Inf. Process.
Assoc. Annu. Summit Conf., 2017, pp. 686–691.

[33] G. Jocher et al., “Ultralytics/yolov5: V3.1 - bug ﬁxes and perfor-
mance improvements,” Oct. 2020. [Online]. Available: https://github.
com/ultralytics/yolov5

[34] F. C. Akyon, E. Akagunduz, S. O. Altinuc, and A. Temizel, “Sequence
models for drone vs bird classiﬁcation,” 2022, arXiv:2207.10409.
[35] C.-Y. Wang, A. Bochkovskiy, and H.-Y. M. Liao, “YOLOv7: Trainable
bag-of-freebies sets new state-of-the-art for real-time object detectors,”
in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2023, pp.
7464–7475.

[36] L. Alan, T. Vojíˇr, L. ˇCehovin, J. Matas, and M. Kristan, “Discriminative
correlation ﬁlter tracker with channel and spatial reliability,” Int. J.
Comput. Vis., vol. 126, no. 7, pp. 671–688, 2018.

[37] S. K. Mistry, S. Chatterjee, A. K. Verma, V. Jakhetiya, B. N. Subudhi,
and S. Jaiswal, “Drone-vs-bird: Drone detection using YOLOv7 with
CSRT tracker,” in Proc. IEEE Int. Conf. Acoust., Speech, Signal Pro-
cess., 2023, pp. 1–2.

[38] G. Jocher et al., “Ultralytics YOLOv8.” 2024. [Online]. Available: https:

//github.com/ultralytics/ultralytics

[39] N. Kim, J.-H. Kim, and C. S. Won, “FAFD: Fast and accurate face

detector,” Electronics, vol. 11, no. 6, 2022, Art. no. 875.

[40] M. Kisantal, Z. Wojna, J. Murawski, J. Naruniec, and K. Cho, “Aug-
mentation for small object detection,” 2019, arXiv:1902.07296.
[41] J.-H. Kim, N. Kim, and C. S. Won, “High-speed drone detection based
on Yolo-V8,” in Proc. IEEE Int. Conf. Acoust., Speech, Signal Process.,
2023, pp. 1–2.

[42] W. Liu et al., “SSD: Single shot multibox detector,” in Proc. 14th Eur.

Conf. Comput. Vis., 2016, pp. 21–37.

[43] P. Dong, C. Wang, Z. Lu, K. Zhang, W. Wan, and J. Sun, “S-feature
pyramid network and attention model for drone detection,” in Proc.
IEEE Int. Conf. Acoust., Speech, Signal Process., 2023, pp. 1–2.

[44] “COCO evaluation toolbox.” 2024.
cocodataset.org/#detection-eval

[Online]. Available: https://

ANGELO COLUCCIA (Senior Member, IEEE) re-
ceived the Ph.D. degree in information engineering
from the University of Salento, Lecce, Italy, in
2011. He is currently an Associate Professor of
Telecommunications at the Department of Engi-
neering, University of Salento. He was a Research
Fellow with Forschungszentrum Telekommunika-
tion Wien, Vienna, Austria, and has held a visiting
position with the Department of Electronics, Op-
tronics, and Signals of the Institut Supérieur de
l’Aéronautique et de l’Espace, Toulouse, France.
His research interests include multi-channel, multi-sensor, and multi-agent
statistical signal processing for detection, estimation, localization, and learn-
ing problems. Relevant application ﬁelds are radar, wireless networks (includ-
ing 5G and beyond), and emerging network contexts (including intelligent
cyber-physical systems, smart devices, and social networks). He is Member
of the Sensor Array and Multichannel Technical Committee and of the Data
Science Initiative for the IEEE Signal Processing Society.

778

VOLUME 5, 2024

ALESSIO FASCISTA (Member, IEEE) received
the Ph.D. degree in engineering of complex sys-
tems from the University of Salento, Lecce, Italy,
in 2019. He has held a visiting position with the
Department of Telecommunications and Systems
Engineering, Universitat Autonoma de Barcelona,
Spain, in 2018, and with the Department of Electri-
cal Engineering, Chalmers University of Technol-
ogy, Gothenburg, Sweden, in 2022. He is currently
an Assistant Professor of telecommunications with
the Department of Innovation Engineering, Univer-
sity of Salento. His main research interests include telecommunications with
focus on statistical signal processing for detection, estimation, and localiza-
tion in terrestrial wireless systems. He is a Member of the Technical Area
Committee in Signal Processing for Multisensor Systems of EURASIP. He
is an Associate Editor for IEEE OPEN JOURNAL OF THE COMMUNICATIONS
SOCIETY (OJ-COMS).

ANASTASIOS DIMOU received the Diploma
in electrical and computer engineering from the
Aristotle University of Thessaloniki, Thessaloniki,
Greece, the Professional Doctorate in engineer-
ing (PDEng) in information and communication
technology from the Technical University of Eind-
the Netherlands, and the Ph.D.
hoven (TU/e),
degree in image processing for surveillance appli-
cations from the Universidad Politcnica de Madrid
(UPM), Spain. He is currently a Researcher with
the Information Technologies Institute (ITI) of the
Centre for Research and Technology Hellas, Greece. His work has led to the
co-authoring of more than 40 publications in refereed international journals
and conferences, including eight journals, 32 conferences, and two book
chapters. He regularly acts as a reviewer for multiple conferences and journals
in his domain.

LARS SOMMER received the B.S. and M.S.
degrees in electrical engineering and information
technology from the Karlsruhe Institute of Tech-
nology (KIT), Karlsruhe, Germany, in 2011 and
2014, respectively, and the Ph.D. degree from
KIT. He is currently a Research Scientist with the
Video Exploitation Systems Department, Fraun-
hofer IOSB. His research mainly include image
analysis using machine learning, especially deep
learning based classiﬁcation, detection and seg-
mentation and explainable AI.

ARNE SCHUMANN received the Diploma in
computer science from the Karlsruhe Institute of
Technology (KIT), Karlsruhe, Germany, in 2011,
and the the Ph.D. from KIT, in 2019. He is cur-
rently a Senior Scientist with the Video Exploita-
tion Systems Department at Fraunhofer IOSB.
He was a Researcher on several computer vi-
sion subjects with KIT, the Fraunhofer Institute
of Optronics, System Technologies and Image Ex-
ploitation (IOSB), Karlsruhe, Germany, and Queen
Mary University of London, London, U.K. He has
authored or coauthored over 45 scientiﬁc publications, several of which focus
on the subject of UAV detection, tracking and classiﬁcation. His work primar-
ily focuses on deep learning methods for image exploitation, including object
detection, classiﬁcation, few-shot learning and data-centric AI methods.

DIMITRIOS ZARPALAS received the Diploma in
electrical and computer engineer from the Aristotle
University of Thessaloniki, Thessaloniki, Greece,
the M.Sc. degree in computer vision from The
Pennsylvania State University, State College, PA,
USA, and the Ph.D. degree in medical informatics
(School of Medicine, A.U.Th). He is curently a
Principal Researcher (grade B) with the Informa-
tion Technologies Institute (ITI), the Centre for
Research and Technology Hellas, Marousi, Greece.
He has coauthored more than 75 papers in peer re-
viewed international journals, conference proceedings, and books (including
one IEEE Distinguished paper and one IEEE conference best paper award).
His main research interests include 3D/4D computer vision and machine
learning, such as tele-immersion applications: volumetric video, 4D recon-
struction of moving humans, their hologram compression and transmission in
real-time, 3D motion capturing, analysis and evaluation, 3D object recog-
nition and 3D shape descriptor extraction, 3D medical image processing,
shape analysis of anatomical structures, while in the past has also worked
in indexing, search and retrieval and classiﬁcation of 3D objects, proteins and
3D model watermarking.

Open Access funding provided by ‘Università del Salento’ within the CRUI CARE Agreement

VOLUME 5, 2024

779

