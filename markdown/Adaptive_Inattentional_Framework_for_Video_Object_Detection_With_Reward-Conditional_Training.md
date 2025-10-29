Received June 23, 2020, accepted June 28, 2020, date of publication July 1, 2020, date of current version July 20, 2020.

Digital Object Identifier 10.1109/ACCESS.2020.3006191

Adaptive Inattentional Framework for Video
Object Detection With Reward-Conditional
Training

ALEJANDRO RODRIGUEZ-RAMOS , (Graduate Student Member, IEEE),
JAVIER RODRIGUEZ-VAZQUEZ, CARLOS SAMPEDRO , (Member, IEEE),
AND PASCUAL CAMPOY , (Senior Member, IEEE)
Centre for Automation and Robotics, Computer Vision and Aerial Robotics Group, Universidad Politécnica de Madrid (UPM-CSIC), 28006 Madrid, Spain

Corresponding author: Alejandro Rodriguez-Ramos (alejandro.rramos@upm.es)

This work was supported in part by the Spanish Ministry of Economy and Competitivity through the project (Complex Coordinated
Inspection and Security Missions by UAVs in cooperation with UGV) under Grant RTI2018-100847-B-C21, in part by the MIT
International Science and Technology Initiatives (MISTI)-Spain through the project (Drone Autonomy), and in part by the Mohamed Bin
Zayed International Robotics Challenge (MBZIRC) in the year 2020 (MBZIRC 2020 competition).

ABSTRACT Recent object detection studies have been focused on video sequences, mostly due to the
increasing demand of industrial applications. Although single-image architectures achieve remarkable
results in terms of accuracy, they do not take advantage of particular properties of the video sequences
and usually require high parallel computational resources, such as desktop GPUs. In this work, an inat-
tentional framework is proposed, where the object context in video frames is dynamically reused in order
to reduce the computation overhead. The context features corresponding to keyframes are fused into a
synthetic feature map, which is further reﬁned using temporal aggregation with ConvLSTMs. Furthermore,
an inattentional policy has been learned to adaptively balance the accuracy and the amount of context
reused. The inattentional policy has been learned under the reinforcement learning paradigm, and using
our novel reward-conditional training scheme, which allows for policy training over a whole distribution of
reward functions and enables the selection of a unique reward function at inference time. Our framework
shows outstanding results on platforms with reduced parallelization capabilities, such as CPUs, achieving
an average latency reduction up to 2.09×, and obtaining FPS rates similar to their equivalent GPU platform,
at the cost of a 1.11× mAP reduction.

INDEX TERMS Inattention, YOTO, reward-conditional training, deep learning, video object detection,
reinforcement learning, CNN, LSTM, loss-conditional training.

I. INTRODUCTION
Recent advances in image object detection have mainly
focused on the development of Convolutional Neural Net-
work (CNN) architectures [1]–[4] to progressively increase
accuracy or decrease processing times. In this regard, accu-
racy has been the primary concern in the majority of studies
[3], [5]–[7], mostly due to the initial lack of techniques
which precisely capture the variability found in the object
detection task (number of classes, illumination and environ-
mental ambiences, corner cases, etc.). Nevertheless, detec-
tion speed and power consumption are recently becoming

The associate editor coordinating the review of this manuscript and

approving it for publication was Ting Wang

.

key differentiator metrics [1], [8]–[13] as deep learning is
being increasingly deployed in practical applications, where
computing and power resources can be limited. In the context
of systems with reduced computational capabilities, such as
embedded platforms, substantial effort has been carried out
to generate efﬁcient architectures, such as the ones based
on SqueezeNet [14], Mobilenet [15]–[18], or ShufﬂeNet
[19], [20]. Also, other studies target efﬁcient representa-
tions, such as Binarized Neural Networks (BNNs) [21] or
Quantized Neural Networks (QNNs) [22], and low-power
hardware implementations [23]–[25]. Despite signiﬁcant
advances in the ﬁeld,
the ﬁnal objective of processing
high-resolution images in real-time, on embedded systems
and without notable accuracy loss, is still an open problem.

VOLUME 8, 2020

This work is licensed under a Creative Commons Attribution 4.0 License. For more information, see https://creativecommons.org/licenses/by/4.0/

124451

A. Rodriguez-Ramos et al.: Adaptive Inattentional Framework for Video Object Detection With Reward-Conditional Training

However, human vision system provides intuition on how
such solution can be achievable. On this basis, human
vision has been found as foveal and active [26], [27]. Envi-
ronment observation at high-resolution is achieved by the
fovea, and only corresponds to 5 degrees of the total visual
ﬁeld [28]. Our complementary peripheral vision, which pro-
cesses observations at a lower resolution, has a ﬁeld of
view of 110 degrees [29]. To track objects, our eyes per-
form quick saccadic movements, followed by smooth pursuit
movements [30], [31]. In this context, when the focus is on
an object, its peripheral surroundings can be simply ignored,
a phenomenon known as inattentional blindness [32]. Hence,
detecting and tracking objects is an active process, and the
amount of information processed depends on the complexity
and the dynamism of the scene, as well as on the attention
of the observer. Besides, a fundamental advantage of human
vision is that it relies on a stream of images, rather than on
single images. Humans can rely on contextual cues and mem-
ory to supplement their understanding of the image. On this
subject, some of the stated properties of human vision have
led to novel neuromorphic sensor designs, such as Dynamic
Vision Sensors (DVS), which react to events and decrease
the amount of information processed. Although DVS-related
research studies are still in early stages, they have provided
promising results in the context of asynchronous object detec-
tion and tracking at high-speeds and low-power consumption
[33], [34]. Conversely, this work, where a standard RGB cam-
era sensor has been utilized, follows the idea of decreasing the
amount of data to be processed, and examines the question
of whether neural networks are similarly able to dynamically
neglect image information when assisted by a memory and
previous image context, providing a lower computation over-
head.

An important property of video is that adjacent video
frames are highly correlated, which opens up the possibility
of decreasing the computation latency. During the process
of object detection in a sequence, surrounding context of
previous frame detection is prone to hold redundant con-
tent, which can be exploited in the current detection stage.
Therefore, a simple idea is to keep a memory of previous
extracted features and recompute only the ones corresponding
to the region where the object was found in previous frame.
To follow this idea, features from previous and current frame
have to be fused in a synthetic feature map, which is used
to compute the ﬁnal detection. In this trend, based on the
intuition that human peripheral vision may not add value
while pursuing an object, our approach dynamically reuses
feature context from previously detected frames to increase
efﬁciency.

From these observations, we propose a simple but effec-
tive pipeline for efﬁcient video object detection, illustrated
in Fig. 1. Concretely, we introduce a novel inattentional
framework, where peripheral detection context is dynami-
cally reused to speed up inference latency while maintaining
accuracy. The context features corresponding to keyframes
are fused into a synthetic feature map, which is further reﬁned

FIGURE 1. A schematic diagram of the proposed inattentional framework.
The keyframe context is dynamically reused to reduce computation
overhead, and temporal information is propagated by the temporal
aggregator. Grey region in an image frame denotes pixels which are not
being processed for the given frame.

based on its temporal structure. The temporal aggregation
is carried out by a Convolutional Long-Short Term Mem-
ory (ConvLSTM) layer, and detections are generated by fus-
ing context from previous frames with the new information of
the current frame. Additionally, we show that the aggregation
of our synthetic feature maps by the ConvLSTM contains
within itself the information necessary to decide when the
peripheral context has to be ignored. We learn an inattentional
policy of when to use the full or synthetic feature map by
formulating the task as a reinforcement learning problem.

In addition, the amount of attention a human observer gives
to an object in a video sequence can be dynamic, modulat-
ing the inattentional blindness based on the one individual
intentions. On the side of deep learning, neural networks are
static structures which response commonly remains invariant
during inference. This is due to the fact that during com-
mon network training stages, the parameters of a given loss
function are not normally altered, narrowing the behavior of
the network to a unique state of the potential distribution of
loss functions. In this regard, recent advances reveal that a
neural network can be trained not only over a unique loss
function but over a distribution of them, as in You Only
Train Once (YOTO) [35], [36]. Following this direction, the
present work explores the possibility of extending this idea
to the reinforcement learning paradigm, where the standard
policy training involves deﬁning a unique performance metric
[37]–[39]. Under the context of the proposed application,
the behavior of the inattentional policy network has been
extended to dynamically provide different attention ratios,
based on an inattentional parameter, which modulates the
target performance metric on inference time.

While prior works, mostly ﬂow [40]–[47] and recurrent
[43], [48]–[52] methods, also provide approaches for fast
video object detection with fast and slow detection stages,
these approaches are based on processing the full image

124452

VOLUME 8, 2020

A. Rodriguez-Ramos et al.: Adaptive Inattentional Framework for Video Object Detection With Reward-Conditional Training

for every frame, incurring in a redundancy of computation
and not exploiting the performance increase when the object
appears reduced in the image. Conversely, our method takes
advantage of the redundancy of context in video frames,
increasing the performance when objects appear small in the
image. Furthermore, approaches based on the reinforcement
learning framework [49], provide policies trained over a
unique performance metric i.e. reward function. Our policy
network has been trained over a distribution of reward func-
tions, providing not only a unique behavior but a distribution
of them on inference time. Our method has been validated
on the Imagenet VID 2015 [53] dataset and on our custom
dataset, which has been made publicly available.

In summary, the contributions of the present work are as

follows:

• We present a human-inspired inattentional framework
for video object detection, where context features are
dynamically reused in a synthetic feature map in order
to reduce redundant computation, and their outputs are
fused using a recurrent memory module.

• We introduce an adaptive inattentional policy where
the decision over the context features computation is
learned with deep reinforcement learning, which leads
to a higher speed-accuracy trade-off.

• We demonstrate a successful application of YOTO [35]
to the reinforcement learning paradigm, where a policy
has been trained over a distribution of reward func-
tions. To the authors knowledge, this is the ﬁrst time
this technique is applied to the reinforcement learning
framework.

• The custom dataset used for validation of the framework

has been made publicly available.

The remainder of the paper is organized as follows:
Section III describes our approach and provides a detailed
explanation about the inattentional fundamentals and the
reward-conditional training, explaining the problem formu-
lation and the system architecture. Section IV-B outlines
the carried-out experiments and their corresponding results
and Section V remarks on and discusses the most relevant
experimentation outcomes. Finally, Section VI concludes the
paper and indicates future lines of research.

II. RELATED WORK
This work proposes a video object detection framework and,
to this aim, is aided by techniques of diverse nature, such as
temporal aggregation or keyframe selection methods. As a
consequence, there are several related works that are adjacent
to the case under study.

A. VIDEO POST-PROCESSING METHODS
Early research for extending single-image object detection to
the video domain was commonly focused on the generation
of frame detection tracks, where current detection is linked to
the previous ones in the track. Through these tracks, previous

information is aggregated to improve the current frame object
detection accuracy.

The Seq-nms algorithm proposed in [54] applies dynamic
programming to ﬁnd tracks and improve the conﬁdence
of weaker detections. Tubelets with Convolutional Neural
Netowrks (TCNN) [55], [56] proposes a pipeline for detec-
tion propagation across frames via optical ﬂow, and a tracking
algorithm to reﬁne scoring by ﬁnding tubelets. These initial
strategies yielded accountable improvements in performance,
but did not fundamentally change the basic per-frame detec-
tion techniques. In our work, we take advantage of the key
idea of aiding current detection with previous detection con-
text to decrease the computation time, while maintaining the
accuracy.

B. FEATURE AGGREGATION OVER TIME
One key aspect which differentiates video from single-image
is the existence of encoded information that
detection,
remains low-variant across several video frames. Conse-
quently, there exist inter-frame features which can be aggre-
gated and exploited to improve performance. In this regard,
multiple techniques for stated feature aggregation have been
explored in the literature.

In [40], intermmediate feature maps in a CNN were able to
be temporally propagated across frames by means of optical
ﬂow. The Deep Feature Flow (DFF) framework [40] allows
for feature propagation across multiple frames in order to
compute detections on sparse keyframes, which alleviates
the computational cost. Flow-Guided Feature Aggregation
(FGFA) [41] further explored this idea by warping and aver-
aging features from adjacent frames, which lead to acurracy
improvements. Impression networks [42] provided a compu-
tation reduction by combining sparse ‘‘impression features’’,
which stores low-variant and long-term information, with
optical ﬂow and warping aggregation. In [43] an efﬁcient
feature ﬂow aggregation for different keyframe selection
schemes is introduced. The Flow-Track framework [44] com-
bines historical feature ﬂows, warping, consine similarity and
temporal attention for efﬁcient feature aggregation achiev-
ing an increased performance. Some techniques, primarily
meant for semantic object segmentation [45]–[47], also uti-
lized warped feature ﬂow maps to aid the ﬁnal stages of the
segmentation pipeline.

Recurrent architectures, such as Gated Recurrent Units
(GRUs) or LSTM cells, have been integrated to enable the
process of feature aggregation. In [43], a ﬂow-guided GRU
is proposed to effectively aggregate features on keyframes.
Spatial-Temporal Memory Modules (STMMs) [48] were pro-
posed to make better use of single-image detector weights
pre-trained on large scale image datasets. ConvLSTMs vari-
ants have been more widely utilized for feature aggrega-
tion across frames [49]–[52]. Although ConvLSTM can
fundamentally serve as a memory which is able to store
mid-term information across frames [50], [51], it can be
further extended to aggregate features from different extrac-
tors [49] or from an encoding of multiple frames [52].

VOLUME 8, 2020

124453

A. Rodriguez-Ramos et al.: Adaptive Inattentional Framework for Video Object Detection With Reward-Conditional Training

Our ConvLSTM memory module is similar to [51], but serves
an additional purpose of fusing features from real and syn-
thetic feature maps, posing an additional challenge.

Furthermore, external memories can beneﬁt the long-term
information storage, which can be useful for feature aggrega-
tion in the video domain [57], [58]. Besides, some techniques
integrate detection trackers to exploit temporal information
between keyframe processing [59]–[61]. Other strategies
rely on Locally-Weighted Deformable Neighbors (LWDNs)
[62], Spatio-Temporal Sampling Networks (STSNs) [63],
Motion History Image (MHI)
spatially variant
convolutions [65] or cross-correlations [66] for feature
propagation.

[64],

C. ADAPTIVE KEYFRAME SELECTION
Multiple approaches interleave processing pipelines with dif-
ferent computational budgets during the execution of the
video object detection, in order to provide both fast and accu-
rate solutions. This procedure commonly involves keyframe
selection, where stated keyframe is processed by a more
expensive but accurate pipeline. Despite keyframe selection
can be carried out naively (i.e. ﬁx rate), providing proper
solutions [40], [43], [66], a more adaptive method for the
case under study is potentially able to improve performance.
For instance, to adapt to video input sources of variable
complexity.

Other

take advantage of

In [61], the optimal combination of detection and tracking
stages is found in an ofﬂine trend for a speciﬁc video. In [43],
a feature consistency indicator was used in a heuristic rule
to select keyframes. A keyframe selection policy has been
also incorporated to standard supervised learning schemes,
such as the usage of low-level features for predicting rapid
changes in the input [65], increase of complexity [45] or
features quality [42].
techniques

the reinforce-
ment learning framework for adaptive keyframe selection
[47], [49], [60]. A policy-gradient reinforcement-learning
approach [47] makes budget-aware processing by approx-
imating the gradient of a non-decomposable and non-
differentiable objective. In [67], a learned policy is able
to select when to update or to initiate a tracker. In [60],
the reinforcement learning policy is guided to achieve a
proper balance between detection and tracking. In [49] a
light policy is used for balancing the execution pipeline
between an expensive and a cheap feature extractor, achiev-
ing better results as compared to a random baseline. Our
keyframe selection strategy has been inspired by the adaptive
keyframe selection method found in [49]. In this direc-
tion, an adaptive selection policy has been learned within
the reinforcement learning framework in order to select
between real and synthetic feature maps (expensive and
cheap pipeline, respectively) for detection. Furthermore, our
complete policy has been trained over a distribution of reward
functions, instead of a unique and constrained reward func-
tion, achieving proper performance throughout the whole
domain.

D. SINGLE-IMAGE RoI SELECTION METHODS
A feed of video images opens the possibility of a potential
increment of efﬁciency due to information redundancy across
frames. Nevertheless, within the single-image context, there
also exist regions which do not contain useful information
to be exploited during the object detection task. In this
regard, a notable amount of techniques have been explored
in the literature. Stated strategies primarily aim at reducing
the amount of information processed in an single image by
selecting Regions of Interest (RoIs).

The generation of saliency maps can help during the RoI
selection procedure. In the context of classical computer
vision, log-spectrum [68], as well as a combination of image
ﬁlters [69] (color, intensity, and orientation ﬁlters, among
others) have been utilized for saliency computation. Addi-
tionally, RoI selection can be automated with standard super-
vised learning techniques. AutoFocus [70] forwards the full
image through a cheap network to generate ‘‘chips’’ which
are further processed in higher resolutions. In [71], a cropped
image and its context is used in order to improve small object
detection. AZ-Net [72] method zooms recursively into RoIs,
which are likely to contain objects. In [73], RoIs are generated
throughout an iterative process by the network, and a voting
process is carried out to select the ﬁnal regression. In [74],
semantic and context information is used to ﬁnd RoIs.

On the other hand, reinforcement learning has been notably
applied to the problem of RoI selection. Dynamic zoom-in
network [9] inputs a full low-resolution image and uses a
reinforcement learning policy to select RoI to be further ﬁne
processed. In [75], an LSTM-based model is trained with
reinforcement learning to give attention to certain RoI of the
image in order to ﬁnd small objects. In [76], [77], a reinforce-
ment learning based sliding window approach ﬁnds an object
in a few steps. Reinforcement learning based Region Proposal
Network (RPN) [10] has also been explored to increase its
efﬁciency of computation. In [78], [79], an agent selects the
regions of the image which need to be processed at higher res-
olution or by a ﬁner detector to reduce computation. In [80]
RoIs are effectively selected based on the interdependence of
objects with tree-structured reinforcement learning. In this
work, RoIs are strongly related to the previous keyframe
detection at a certain instant of time. Also, RoI size decision
is based on the state of our ConvLSTM, being potentially
assisted by the context of previous detected object in the
sequence.

III. THE INATTENTIONAL FRAMEWORK
In this section, the proposed framework is described. First,
the inattentional fundamentals are explained. Then, the adap-
tive inattentional policy as well as the reward-conditional
training procedure are described in detail. The nomenclature
and abbreviations used in the description of the inattentional
framework are summarized in Table 1.

A. INATTENTIONAL FUNDAMENTALS
Following the intuition provided by the inattentional blind-
ness phenomenon [32], our solution is approached as a

124454

VOLUME 8, 2020

A. Rodriguez-Ramos et al.: Adaptive Inattentional Framework for Video Object Detection With Reward-Conditional Training

FIGURE 2. Structure of the extended data augmentation technique. The architecture shown corresponds to MobileNetV1 feature extractor. The
receptive field in a deep layer of a standard CNN is usually a notable region of the input image. In order to further reduce the computation overhead,
image crops I ∗
have not been selected to match the required receptive field. To tackle this issue, the resulting feature maps Ck have been included in
k
the augmented training data for generality of the temporal aggregator.

TABLE 1. Table of abbreviations and nomenclature in the context of the
proposed inattentional framework.

dynamic context reuse across an image sequence V =
{I0, I1, I2, . . . , In}. Indeed, the phenomenon of momentarily
ignoring context, while detecting an object, can be effectively
reformulated as a context reuse, since reusing previous con-
text is equivalent to avoiding current context computation.
Besides, our framework is restricted to the online setting
where only {I0, I1, . . . , Ik } are available during the compu-
tation of the k-th detection.

In order to materialize this concept into the deep learning
paradigm, four main components have been deﬁned in the
present framework: a feature extractor f, a context aggregator
c, a temporal aggregator a and a detector d. The context
is deﬁned as a subset of the feature maps generated by the
feature extractor f on a keyframe k. In this direction, our
approach interleaves full keyframe computation and partial
frame (inattentional frame) computation with context aggre-
gation (see Fig. 2 and Fig. 3) to provide an intermmediate
representation. The frame-level features are then temporally
aggregated and reﬁned using a recurrent architecture. Finally,
an SSD-style [17] detection pipeline is applied on the reﬁned
feature map to obtain bounding box results.

FIGURE 3. A block diagram of our adaptive keyframe selection technique
with all the components of the inattentional framework included, where f
stands for feature extractor, c is the context aggregator, a depicts the
temporal aggregator and d represents the detector. For clarity, coloured
blocks correspond to tensors and grey blocks correspond to components
of the system. The length of the tensor denotes time consumption but it
is not to scale. The context of a keyframe F c
k
throughout the pipeline of context aggregators.

is not being modified

Each step of the processing pipeline can be deﬁned as a
function mapping. The feature extractor f: RI → RF , maps
the image space into an encoded feature space RF . The con-
text aggregator c: RF ×RF → RC optionally composes a syn-
thetic feature map Ck , based on previous keyframe context F c
k
and current reduced feature map F ∗
k . The temporal aggregator
a: RF × RS → RA × RS or RC × RS → RA × RS , reﬁnes the
generated feature map (either full RF or synthetic RC ) based
on previous temporal information cues, compute its state RS

VOLUME 8, 2020

124455

A. Rodriguez-Ramos et al.: Adaptive Inattentional Framework for Video Object Detection With Reward-Conditional Training

and outputs an updated feature space RA. The SSD detector
d: RA → RD, maps the aggregated feature space into ﬁnal
detection predictions. In order to obtain detections Dk , one
may compute Ak by a(f(Ik ), sk−1) or a(c(f(I ∗
k ), sk−1),
k ), F c
where sk−1 is the previous frame state and, Ik and I ∗
k are the
full image or cropped image for the k-th frame, respectively.
The ﬁnal detections Dk are obtained as Dk = d(Ak ).

The context aggregator c is a key component in the inatten-
tional framework, as it is in charge of reusing context from
previous keyframes, in order to provide a more efﬁcient per-
formance while reducing computation overhead. Speciﬁcally,
reusing context requires fusing the current feature map F ∗
k
with the context feature map F c
k . Since the context is related
to the object in the image space, one simple approach for
fusing feature maps is to calculate the precise position of
a feature map corresponding to an image crop I ∗
k inside a
context feature map F c
k corresponding to a full keyframe Ik−n.
Furthermore, in order to precisely select the context crop in
the image space (image crop I ∗
k ), previous detection Dk−1
and the particular receptive ﬁeld of Fk have to be taken into
consideration. However, the receptive ﬁeld of a given layer
unit is crucial to generate a feature map F ∗
k which can be
directly placed (without distortion) in the context feature map
F c
k . In this respect, it is important to highlight that the recep-
tive ﬁeld size of a deep layer in a neural network normally
covers a notable percentage of the input image (the receptive
ﬁeld size of the ﬁnal layer increases with the depth and the
striding of the network). At this point, a trade-off, between the
image crop I ∗
k size and the computation overhead stands out,
since in order to account the full size of the receptive ﬁeld, the
image crop can be increased for a certain object, incurring in
more computation overhead, and vice versa.

As the present framework targets computation efﬁciency,
the amount of aggregated context has to be maximized in
order to reduce the computation overhead. This fact leads
to the generation of image crops I ∗
k with a small size which
may not be matching the required receptive ﬁeld for the
feature map Fk (see Fig. 2). In this scenario, the aggregated
feature map Ck can exhibit artifacts around the area where
F ∗
k was placed, resulting in a lower accuracy of the complete
approach. To tackle this issue, the temporal aggregator a has
been also trained to adapt RC → RF domains via extended
feature map augmentation during training. Thus, the temporal
aggregator has been trained with augmented data which aids
the detection with synthetic feature maps (refer to Training
Methodology in Section IV-A). In this context, the temporal
aggregator not only aggregates temporal video information,
but also reﬁnes feature maps from different nature (full and
synthetic).

complexity of the input as information sources. In this regard,
a variable amount of inattention can be provided to the con-
text, based on the uncertainty of past detections and the object
dynamism in a given sequence. Accordingly, a novel adaptive
inattentional policy has been proposed, using formulation of
the reinforcement learning paradigm.

In reinforcement learning, an agent interacts with an envi-
ronment, seeking to ﬁnd the maximum accumulated reward
over time. To formulate the reinforcement learning problem,
it is necessary to deﬁne an action space A, a state space S,
and a reward function r. In the proposed inattentional frame-
work, the inattentional policy has the ability of executing two
actions: it can decide whether to execute the full expensive
pipeline or, on the contrary, to select the faster context
aggregation pipeline (with the potential loss on accuracy).
In this regard, the discrete action space A is deﬁned as A = a
with a ∈ {0, 1}.

Additionally, our proposed policy is meant to examine the
temporal aggregator state in order to ﬁnd insights about the
uncertainty of the detection, as well as about abrupt context
changes in order to perform the best possible action, leading
to an optimum policy π ∗(st |θ). Following this idea, the state
space S is deﬁned as

S = (ck , hk , ck − ck−1, hk − hk−1, ρk , bk−1, ψ)

(1)

where ck and ck−1 are the current and previous ConvLSTM
cell states, respectively; hk and hk−1 are the current and
previous ConvLSTM hidden states, respectively; ρk is an
action history vector and has been empirically found to work
properly with size 20, bk−1 is the normalized previous I ∗
k RoI
(it is the normalized full image RoI if previous frame was
a keyframe), and ψ is the inattentional factor. The inclusion
of (ck − ck−1) and (hk − hk−1) helps detecting changes in
the temporal aggregator state. Also, ρk and bk−1 make the
agent aware of its previous actions and context size, in order
to avoid excessively running the context aggregation pipeline.
The size of the state with the current architecture is 102,425
continuous variables.

The reward function r is a crucial component in the rein-
forcement learning framework. Indeed, it can either speed up
training or, conversely, a naive design can completely prevent
the agent from learning. Our reward must reﬂect the intention
of ﬁnding balance between running the context aggregation
pipeline as much as possible while maintaining accuracy.
In this work, the reward function has been adapted from [49]
as

r =






min
i
ψ + min

i

(L(Di)) − L(D0)

a = 0

(L(Di)) − L(D1) a = 1

(2)

B. ADAPTIVE INATTENTIONAL POLICY
Although context aggregation at random intervals is able
to provide competitive results in terms of latency reduction
while maintaining accuracy (refer to Section IV-B), a nat-
ural question is whether an adaptive policy can improve
these results by using the state of the system as well as the

where D0 and D1 are the detection results through the
expensive pipeline (full image) and cheap pipeline (context
aggregation), respectively; L(·) is the Multibox loss [81]
computation and ψ corresponds to the inattentional factor.
The inattentional factor ψ is an important component of the
reward function. It is a scalar value which can potentially

124456

VOLUME 8, 2020

A. Rodriguez-Ramos et al.: Adaptive Inattentional Framework for Video Object Detection With Reward-Conditional Training

encourage the agent taking the cheap pipeline, even when its
cost remains higher. The deﬁnition of the ψ value is decisive
for the ﬁnal behavior of the agent.

C. REWARD-CONDITIONAL TRAINING
Conventional methods for training machine learning algo-
rithms deﬁne one or more differentiable cost functions to
perform the gradient propagation to the input(s). Typically,
cost functions include static parameters which are able to
modulate training and/or inference performance, leading to
a static behavior of the network at inference time. Stated cost
function parameters can be tuned based on experience, trial-
and-error, grid search or more advanced methods. However,
recent techniques have revealed a convolutional network can
be trained over the whole distribution of parameterized cost
functions, as shown in (3), enabling the possibility of con-
ditioning the network on a subset of parameters at inference
time (YOTO [35]).

θ ∗ = arg min

θ

n
(cid:88)

i=1

L(yi

, F(xi, θ, λi), λi),

λi ∼ Pλ

(3)

where L represent one of the n cost functions, λi are the cost
function(s) parameters and θ the model parameters. Also,
in [35], Feature-wise Linear Modulation (FiLM) [36] is used
to condition the network on the loss parameters. This tech-
nique has been applied in [35] to fully convolutional models
in a supervised and semi-supervised trend, e.g. β-variational
autoencoder, image compression and style transfer.

In this work, we extend the loss-conditional training frame-
work [35] by proposing a novel reward-conditional train-
ing, under the reinforcement learning formulation. As in
the supervised or semi-supervised case, a common agent
in reinforcement learning is trained over a parameterized
performance function, in this case a reward function, which
has to be properly adjusted to encourage the desired behavior.
Once the reward function and its parameters are deﬁned, the
agent behavior converges to a subset of possible behaviors
within the distribution of the reward function parameters.
Instead of deﬁning a unique parameterized reward function,
we propose to condition the network policy π ∗(st |θ, λi) on
the whole distribution of parameterized reward functions,
as in (4) and (5).

π ∗(st |θ, λi) = arg max

Q(st , at |θ, λi),

λi ∼ Pλ

(4)

Q(st , at |θ, λi) = Eπ [Rt (λi)]

at ∈A

∞
(cid:88)

= Eπ [

γ k
r r(λi)t+k+1]

(5)

k=0

where θ corresponds to the model function parameters, λi are
the reward-conditional parameters, γr is the discount factor
and r(λi) denotes the distribution of conditioned reward func-
tions.

For the case under study, our reward function is composed
of one important parameter ψ = λ0, which balances the
amount of accuracy an agent is able to sacriﬁce when running

FIGURE 4. The architecture of the inattentional policy network. The state
tensors have been depicted in dark blue, as well as the λ
reward-conditional parameter. Part of the state (ρ and b) is being
appended in the last fully connected layer of the network. The
reward-conditional parameter has been integrated by applying FiLM
technique to the tensors represented in dark green.

0

both expensive and cheap pipelines. λ0 has been selected as
the reward-conditional parameter. Analogously to the reward
expression shown in (2), the formulation of the ﬁnal condi-
tioned reward function is illustrated in (6) for clarity.

r(λ0) =






min
i
λ0 + min

i

(L(Di)) − L(D0)

a = 0

(L(Di)) − L(D1) a = 1

(6)

where λ0 ∼ Pλ and Pλ denotes a log uniform distribution of
probability with λ0 ∈ [0, 2].

In our experiments, the policy π(st |θ, λ0) is built on a light
convolutional backbone and two Fully Connected Models
(FCMs), which output both the action and the value (see
Fig. 4). To condition the network on the reward parameters,
we use FiLM. We have selected the convolutional layers and
one fully-connected layer to be conditioned by λ0. Assume
a given feature map f of dimensions W × H × C, with W
and H corresponding to the spatial dimensions and C to the
channels. The scalar value λ0 is fed to two FCMs, Mσ and
Mµ, to generate two vectors, σ and µ of dimensionality C
each. We then multiply the feature map channel-wise by σ
and add µ to get the transformed feature map ˆf, as in (7).
ˆfijk = σk ◦ fijk + µk

(7)

where σ = Mσ (λ0) and µ = Mµ(λ0), and ijk correspond to
the feature location within the W × H × C scope. Following
this technique, the policy can be fully learned and conditioned
with λ0 at inference time, providing different attention ratios
while maintaining accuracy. The reward-conditional training
framework has been validated for the challenging application
under study. However, a complete analysis and validation of
the reward-conditional framework is out of the scope of this
work and have been proposed for future work.

D. SYSTEM ARCHITECTURE
The architecture of the system is designed with the aim
of maximizing the computation efﬁciency. MobileNetV1
[16] and MobileNetV2 [17] architectures have been uti-
lized as feature extractors f or backbones. MobileNetV1
and MobileNetV2 backbones have been slightly modiﬁed,

VOLUME 8, 2020

124457

A. Rodriguez-Ramos et al.: Adaptive Inattentional Framework for Video Object Detection With Reward-Conditional Training

to standardize their output size, with respect to the feature
aggregator input size. The conv14 layer in MobileNetV1 has
been removed. In MobileNetV2, the stride has been set to 1 in
the 6-th bottleneck layer, and the number of channels has
been reduced to 512 in the last conv layer (prior to the clas-
siﬁcation stage in the original model). For both backbones,
a depth-multiplier α = 1.0 has been selected for experi-
mentation, which allows for a baseline model capacity to
extract representative features in the context of the presented
datasets. With proper model capacity, the models are less
biased, and the performance of our additional adaptive policy,
whose primary inputs are the stated extracted features, can be
adequately validated. Nevertheless, our framework provides
sufﬁcient generality to use arbitrary feature extractor designs
(α > 0, ResNet variants [82], etc.). The receptive ﬁeld size
of MobileNetV1 and MobileNetV2 for the last layer in each
feature extractor is 219 px × 219 px and 395 px × 395 px,
respectively.

The feature aggregator a is composed of a ConvLSTM
cell, which is in charge of aggregating both time and spa-
tial features. The ConvLSTM cell has been implemented as
in [50]. The bottleneck design of the ConvLSTM and its
depth-wise separable convolutions reduce the computational
costs. This fact is crucial, since the temporal aggregator has
to be executed for every frame of the sequence. The size
of the hidden and cell states of the ConvLSTM have been
adapted based on the input image size. For the detector d,
an SSDLite architecture [17] has been adopted, with sepa-
rable convolutions and depth-wise channel size in all layers.
Finally, the policy network π(st |θ, λ0) has been trained with
Proximal Policy Optimization (PPO) [37] algorithm, which
is an actor-critic algorithm. The architecture has been shown
in Table 2, without accounting the extra FCMs required for
the reward-conditional training for clarity. The FCMs cor-
responding to the reward-conditional training are shallow
fully-connected models with one hidden layer each (vary-
ing from 128 to 512 units) and one output layer (varying
from 64 to 512 units). The layers corresponding to the value
network, which shares weights with the policy network, have
been also represented in Table 2.

IV. EXPERIMENTS AND RESULTS
In the following section, the proposed inattentional frame-
work has been thoroughly tested and validated. The training
methodology section describes the staged training procedure
for the presented system. The results section shows the exper-
imentation outcome corresponding to every component of our
system in a wide variety of scenarios. Finally, the discussion
section interrelates the results to extract insights on our frame-
work design.

A. TRAINING METHODOLOGY
As explained in previous sections, the proposed system is
composed of several components. In this context, the training
is divided into three phases: feature extractor (and detector),
temporal aggregator (and detector) and adaptive inattentional

TABLE 2. The proposed policy network architecture. For clarity, the layers
corresponding to the reward-conditional training FCMs have not been
included. The layers from conv1 to fc1 are shared layers between the
policy and the value models. The fully connected layers are relative to a
320 × 320 input image resolution.

policy training. Although the complete system can poten-
tially be trained end-to-end, we approached a staged training
pipeline in order to avoid convergence issues. The feature
extractor, the temporal aggregator and detector have been
trained in PyTorch [83], while the adaptive inattentional pol-
icy has been trained in Tensorﬂow [84]. The GPU used for
training has been an Nvidia GeForce RTX 2080 Ti. All the
models have been trained on a subset of 260,844 images from
Imagenet VID 2015 dataset [53] and on 29,500 images from a
custom Multirotor Aerial Vehicles VID1 (MAV-VID) dataset,
which has been made publicly available (validation sets are
composed of 3,139 and 10,732 images, respectively). The
Imagenet VID 2015 dataset has been reduced by selecting the
7-th smallest classes in terms of bounding boxes area (varying
from 18% to 30% of the full-size image). The resulting exper-
iments provide the generality of the public Imagenet dataset,
and additionally represent the ultimate goal of the proposed
framework, which is reusing context features when images
context is notable across frames. Furthermore, our custom
MAV-VID dataset constitutes the target application domain
for the case under study, where the object to be detected
appears normally small-sized in the image plane, such as the
case of ﬂying multirotor vehicles.

1) FEATURE EXTRACTOR
In the ﬁrst stage, the feature extractor has been trained in
conjunction with the SSD detector header, in order to extract
robust features to be used by the following stages. Ran-
dom single images from the video sequences and a batch
of 16 images have been used. Adam has been selected as
the optimizer, with a learning rate of 10-4. We use an input
resolution of 320 × 320 and reduce on plateau learning rate
scheduler. We include hard negative mining and data augmen-
tation as described in [81]. The original hard negative mining
approach has been adjusted by allowing a ratio of 10 negative
examples for each positive while scaling each negative loss by

1https://www.kaggle.com/alejodosr/multirotor-aerial-vehicle-vid-

mavvid-dataset

124458

VOLUME 8, 2020

A. Rodriguez-Ramos et al.: Adaptive Inattentional Framework for Video Object Detection With Reward-Conditional Training

0.3. The validation phase is carried out through the whole set
of validation sequences of the datasets.

2) TEMPORAL AGGREGATOR
In the second stage, the feature extractor weights have been
frozen during training. For MobileNetV1, the conv14 layer
has been removed in order to inject the ConvLSTM layer.
Again, the temporal aggregator is trained along with the
SSD detection headers. We sample random sequences (of 10
frames each) from the training set for training. We unroll
the LSTM to 10 frames in which we apply the same data
augmentation techniques as in previous stage. The same
data augmentation transformation is applied to every frame
of each sequence in order to avoid missing the correlation
between consecutive frames. We use a batch of 50 sequences
and an image input size of 320 × 320.

Additionally, an extended data augmentation technique has
been included to account for the structure of the synthetic
feature maps. As explained in Section III-A, due to the proce-
dure of context reuse, where a reduced feature map is placed
in a context feature map, artifacts can appear close to the
placement volume. In order to tackle this effect, during train-
ing, the feature map which results from the feature extraction
stage is randomly modiﬁed with the ground-truth context of
previous frame. We set a 0.01 probability for a feature map
to be modiﬁed by this technique. Adam optimizer has been
utilized with a learning rate of 10−4 and reduce on plateau
learning rate scheduler. The validation is carried out through
the validation sequences in each dataset.

3) ADAPTIVE INATTENTIONAL POLICY
The adaptive inattetional policy has been trained using PPO
algorithm. PPO is a sample-efﬁcient and actor-critic deep
reinforcement learning algorithm. A clip range of 0.1, tra-
jectories of 128 steps and minibatches of 4 elements have
been utilized. The policy network has been trained with Adam
and a learning rate of 2.5 · 10−4 (refer to the code repository
Supplementary Material section for remaining hyperparam-
eters deﬁnition). The observation state has been normalized
in the environment based on its mean and variance through
time (at a certain number of steps, normalization parameters
calculation is stopped). The reward has not been normalized,
since its original design is meant for relative performance
signaling. λ0 parameter has been sampled from a log-uniform
distribution every 10 sequences. Training has been carried out
with the validation set of each dataset. In order to provide
enough variability and avoid overﬁtting, a random transform
as in [81] has been applied to every sequence. During testing,
the random transform is not applied to allow for comparison.

B. RESULTS
For all results, the standard Imagenet VID accuracy metric
is reported, mean Average Precision mAP@0.5 IOU. The
latency of the networks in milliseconds (ms) and the average
Frames per Second (FPS) of the complete approach is pro-
vided. The number of parameters is reported as benchmarks

for efﬁciency. Also, the power consumption in Watts (W)
and the energy efﬁciency in FPS/W are included for every
experiment, as reported in [11]–[13].

The validation tests have been performed in two differ-
ent GPUs, an Nvidia GeForce RTX 2080 Ti desktop GPU
and an Nvidia Volta embedded GPU inside an Nvidia Jet-
son AGX Xavier platform. Also, since the latency reduc-
tion greatly stands out when the parallelization capacity is
decreased, such as the case of desktop and embedded CPUs,
the proposed system has been further tested and validated
in an Intel Core i7-9700K@3.60GHz desktop CPU and a
ARMv8.2@1.377GHz embedded CPU. All the tests have
been performed in Python 3.6 and Ubuntu 18.04 operating
system. The code has been made publicly available (see
Supplementary Material).

In Table 3 and Table 4,

the results corresponding
to MobileNetV1 feature extractor
for both Imagenet
VID 2015 and MAV-VID dataset are shown. In addi-
tion,
the results correspond-
ing to MobileNetV2 feature extractor for both Imagenet
VID 2015 and MAV-VID dataset are also depicted.

in Table 5 and Table 6,

In order to further validate the performance of the pro-
posed inattentional policy, a comparison with a random base-
line for a wide variety of scenarios has been carried out.
In Fig. 5 the trade-off between the number of inattentional
frames executed and the resulting mAP is illustrated for
both datasets and feature extractors. For every value of the
reward-conditional parameter λ0, there is an overall percent-
age of inattentional frames executed. In order to be able to
compare it to a random baseline, a random policy has been
executed the same exact amount of inattentional frames. Also,
the original mAP (with no inattentional frames involved) is
included. Every test has been performed 5 times each, for both
policies, and the average results have been plotted.

Finally, four application cases have been illustrated in
Fig. 6. Two complex scenarios, due to object high-motion
speed within the image plane or environmental complexity
(camera pointing to the sun); and two simple (or static) sce-
narios, where the object stays almost static within the image
plane, have been additionally depicted.

V. DISCUSSION
The thorough testing and validation of the proposed inat-
tentional system describes a proper framework to adaptively
reduce computation overhead in the context of video object
detection. A wide variety of scenarios have been tested for
two feature extractors (MobileNetV1 and MobileNetV2) and
two datasets (Imagenet VID 2015 and MAV-VID). Regarding
Table 3, 4, 5 and 6, both architectures provide state-of-the-art
mAP@0.5IOU in the Imagenet VID 2015 dataset, resulting
in 0.5236 and 0.4924 for MobileNetV1 and MobileNetV2
feature extractors, respectively. Also, the mAP results in
our custom MAV-VID dataset are notable, with 0.9398 and
0.9453 for MobileNetV1 and MobileNetV2 feature extrac-
tors, respectively. MobileNetV1 provided higher mAP in

VOLUME 8, 2020

124459

A. Rodriguez-Ramos et al.: Adaptive Inattentional Framework for Video Object Detection With Reward-Conditional Training

TABLE 3. Results using the MobileNetV1 feature extractor on the ImageNet VID 2015 dataset. The mean Average Precision (mAP), KeyFrame (KF) and
Inattentional Frame (IF) latency, as well as the average runtime FPS are provided. Also, the number of parameters of the models has been included. The
FPS ratio with respect to the base FPS value (without the inattentional policy) has been also included for clarity. Additionally, power consumption and
energy efficiency have been provided for every experiment.

TABLE 4. Results using the MobileNetV1 feature extractor on the MAV-VID dataset. The mean Average Precision (mAP), KeyFrame (KF) and Inattentional
Frame (IF) latency, as well as the average runtime FPS are provided. The FPS ratio with respect to the base FPS value (without the inattentional policy) has
been also included for clarity. Additionally, power consumption and energy efficiency have been provided for every experiment.

Imagenet VID 2015 dataset, whereas MobileNetV2 yielded
higher mAP in the MAV-VID dataset.

The mAP gets reduced when the reward-conditional
parameter λ0 increases, since the policy is encouraged to
execute more inattentional frames at the cost of accuracy.
Nevertheless, it maintains competitive values throughout the
whole range, with a minimum accuracy of 0.4373 and 0.8403

for Imagenet VID 2015 and MAV-VID dataset, respectively.
As shown in Table 5 and 6, MobileNetV2-ConvLSTM-
SSDLite has remained as the model with lower mAP
degradation.

The proposed inattentional framework has achieved a
considerable latency reduction, increasing the runtime FPS
in every platform tested, with minimal accuracy drop.

124460

VOLUME 8, 2020

A. Rodriguez-Ramos et al.: Adaptive Inattentional Framework for Video Object Detection With Reward-Conditional Training

TABLE 5. Results using the MobileNetV2 feature extractor on the ImageNet VID 2015 dataset. The mean Average Precision (mAP), KeyFrame (KF) and
Inattentional Frame (IF) latency, as well as the average runtime FPS are provided. Also, the number of parameters of the models has been included. The
FPS ratio with respect to the base FPS value (without the inattentional policy) has been also included for clarity. Additionally, power consumption and
energy efficiency have been provided for every experiment.

TABLE 6. Results using the MobileNetV2 feature extractor on the MAV-VID dataset. The mean Average Precision (mAP), KeyFrame (KF) and Inattentional
Frame (IF) latency, as well as the average runtime FPS are provided. The FPS ratio with respect to the base FPS value (without the inattentional policy) has
been also included for clarity. Additionally, power consumption and energy efficiency have been provided for every experiment.

Nevertheless, the amount of computation reduction is highly
dependent on the platform where the system is executed,
as well as on the average object size within the image plane.
Considering the GPU platforms, with a higher paralleliza-
tion capacity, the average FPS increase ratio has ranged
from 1.0 to 1.14 for MobileNetV1 and from 1.0 to 1.07

for MobileNetV2. However, regarding the CPU platforms,
where the parallelization capacity is limited, the average FPS
increase ratio has ranged from 1.0 to 2.09 for MobileNetV1
and from 1.0 to 2.06 for MobileNetV2. These results lead
to an average FPS on the desktop CPU platform of 37.73,
which is in the order of magnitude of the base runtime

VOLUME 8, 2020

124461

A. Rodriguez-Ramos et al.: Adaptive Inattentional Framework for Video Object Detection With Reward-Conditional Training

FIGURE 5. Representation of the trade-off between the mAP and the percentage of inattentional frames executed. Four
cases with two feature extractors (MobileNetV1 and MobileNetV2) and two datasets (Imagenet VID 2015 and MAV-VID)
have been evaluated. The random baseline policy is depicted in pale blue (circles) and the proposed inattentional policy
in pale green (triangles). The original mAP (with no inattentional frames involved) is represented with a grey dashed line.
The inattentional policy predominates allover the tests, resulting in a higher mAP for every percentage of inattentional
frames. Also, the inattentional policy incurs less accuracy degradation even at extreme ratios of inattentional frames,
suggesting that our method is superior at capturing the temporal dynamics inherent to videos. Every test has been
performed 5 times each, for both policies, and the average results have been plotted.

FPS on the desktop GPU platform of 51.91, at the cost
of 0.086 mAP reduction, for MobileNetV1 in Imagenet
VID 2015 dataset. The maximum FPS increase ratio has
been 2.09 for MobileNetV1 on the desktop CPU, achieving
a runtime FPS of 24.39 at the cost of 0.09 mAP reduction,
in MAV-VID dataset (where the object size is on average
smaller in the image plane). These results suggest that the
inattentional framework is able to increase computation efﬁ-
ciency when the effective parallelization capacity is lim-
ited. The effective parallelization capacity is inﬂuenced by
several variables and it is relative to the input images, the
model size and the platform speciﬁcations. Qualitatively,
parallelization is low in average CPUs, in applications where
the input images are at high resolution or in applications
where the models have notably more parameters than our
MobileNetV1/V2-ConvLSTM-SSDLite models. These facts
yield to an open ﬁeld of research, where the proposed inatten-
tional framework can be applied. Nevertheless, such exten-
sive study is out of the scope of the present work.

Regarding power consumption and energy efﬁciency,
framework provides proper
the proposed inattentional
results in terms of both global power consumption and
energy efﬁciency increase. Power consumption is notably
higher in desktop GPU/CPUs, with on average 35× more
power consumption than the embedded Xavier platform.
The lowest energy consumption has been 2.87 W for
MobileNetV1-ConvLSTM-SSDLite base model (no inatten-
tional policy) in Imagenet VID 2015 dataset. Energy efﬁ-
ciency varies across platforms, being the GPUs more energy

efﬁcient due to their parallelization capabilities. The maxi-
mum energy efﬁciency has resulted in 3.19 FPS/W (1.12×)
for MobileNetV1 on the Xavier GPU, achieving a runtime
FPS of 13.37 at the cost of 4.19 W, in MAV-VID dataset. It has
to be noted that, regarding relative energy efﬁciency increase
with respect to the base energy efﬁciency (no inattentional
frames), the results follows approximately the same ratios as
in the FPS case, resulting in a maximum energy efﬁciency
increase of 2.09× for MobileNetV1 on the desktop CPU, and
achieving a runtime FPS of 24.39 at the cost of 0.09 mAP
reduction, in MAV-VID dataset.

Considering Fig. 5, in comparison to the random base-
line, our inattentional policy predominates allover the tests,
providing a higher mAP for every percentage of inatten-
tional frames executed. Furthermore, our inattentional policy
shows lower mAP degradation even at extreme percentage
of inattentional frames, suggesting that our method is supe-
rior at capturing the temporal dynamics inherent to videos.
The highest mAP distance to the random baseline has been
provided by the MobileNetV1-ConvLSTM-SSDLite in the
Imagenet VID 2015 dataset, with a distance of 0.4 in mAP
at 92% of inattentional frames executed. Another emergent
property is that, for the case of MobileNetV1 in Imagenet
VID 2015 dataset, the inattentional policy is able to match
the original mAP even when executing a 40% of inattentional
frames, which suggests there is a redundancy of information
that does not add value to the ﬁnal accuracy. In addition,
thanks to our novel reward-conditional
training scheme,
the policy can be conditioned at inference time, providing

124462

VOLUME 8, 2020

A. Rodriguez-Ramos et al.: Adaptive Inattentional Framework for Video Object Detection With Reward-Conditional Training

FIGURE 6. Images corresponding to four different scenarios for both datasets. Every image example has been processed by
MobileNetV1-ConvLSTM-SSDLite [λ
0 = 1.2] model (MobileNetV2-ConvLSTM-SSDLite provides a similar behavior). Shaded regions correspond to
non-processed pixels. (a)-(b) and (c)-(d) image examples correspond to Imagenet VID 2015 and MAV-VID dataset, respectively. (a) and (c) show
complex detection examples, corresponding to an object with high-motion speed within the image plane, and to an increased environmental
complexity (camera pointing to the sun), respectively. In this context, the inattentional policy executes a lower percentage of inattentional frames.
(b) and (d) illustrate objects which remain more static throughout the sequence. In this scenario, the number of inattentional frames is higher.

promising results in a wide variety of sequences with varying
backgrounds, changes in illumination, object size, etc. and
allowing for real-time performance modulation in the context
of the required application (see supplementary video2).

Finally, at the cost of 2.2M parameters, a learned inatten-
tional policy can provide adaptiveness to the video dynamics,
as shown in Fig. 6. In this ﬁgure, two complex and two simple
scenarios, in terms of detection difﬁcultness, are illustrated.
In the complex scenarios, where the object is moving fast
within the image plane, or there are ambient conditions which
difﬁcult detection, the inattentional policy performs a higher
rate of full keyframes in order to maintain the detection
accuracy of the object through the sequence. Nevertheless,
when the scenarios are simpler, such as the case an object with
slow motion in the image plane, the inattentional policy tends
to neglect context to speedup computation, without missing
accuracy. This adaptive behavior can be very promising in a
wide variety of video object detection applications.

VI. CONCLUSION
In this work, the inattentional framework has been studied.
The presented framework aims at reducing the computation
overhead, in the frame of video object detection, by reusing
redundant context in video images. An inattentional policy
has been learned, under the reinforcement learning paradigm,
to select the amount of frames where the context is reused.

2https://vimeo.com/426725929

VOLUME 8, 2020

Furthermore, a novel reward-conditional training has been
presented, where a policy can be trained on a distribution
of reward functions and conditioned on one unique function
at inference time. The inattentional framework provided an
average latency reduction in CPUs up to 2.09 times the origi-
nal latency, and obtaining FPS rates similar to their equivalent
GPU platform, at the cost of a mAP reduction of 1.11 times.
This study could be extended by evaluating the inatten-
tional framework in other scenarios of low parallelization
capacity, such as the case of high resolution input images or
mobile devices. Also, optimization techniques and shallower
architectures can be further applied, such as half-precision
inference, quantization or MobileNets with α < 1. Regard-
ing the reward-conditional training, a complete study of this
method with a diverse set of reward functions and reinforce-
ment learning algorithms has been left as future work.

SUPPLEMENTARY MATERIAL
The code has been made publicly available at https://
github.com/alejodosr/adaptive-inattention. Also,
short
video demonstration can be found at https://vimeo.com/
426725929. The MAV-VID dataset can be downloaded at
https://www.kaggle.com/alejodosr/multirotoraerial-vehicle-
vid-mavvid-dataset.

a

ACKNOWLEDGMENT
The authors would like to acknowledge Estefanía Carolina
Asimbaya Shuguli for the exhaustive annotations of the
images for the generation of the public dataset.

124463

A. Rodriguez-Ramos et al.: Adaptive Inattentional Framework for Video Object Detection With Reward-Conditional Training

REFERENCES
[1] A. Bochkovskiy, C.-Y. Wang, and H.-Y. Mark Liao, ‘‘YOLOv4: Opti-
mal speed and accuracy of object detection,’’ 2020, arXiv:2004.10934.
[Online]. Available: http://arxiv.org/abs/2004.10934

[2] S. Liu, D. Huang, and Y. Wang, ‘‘Learning spatial fusion for single-
shot object detection,’’ 2019, arXiv:1911.09516. [Online]. Available:
http://arxiv.org/abs/1911.09516

[3] T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár, ‘‘Focal loss for
dense object detection,’’ in Proc. IEEE Int. Conf. Comput. Vis., Oct. 2017,
pp. 2980–2988.

[4] M. Tan, R. Pang, and Q. V. Le, ‘‘EfﬁcientDet: Scalable and efﬁ-
cient object detection,’’ 2019, arXiv:1911.09070. [Online]. Available:
http://arxiv.org/abs/1911.09070

[5] X. Zhou, D. Wang, and P. Krähenbühl, ‘‘Objects as points,’’ 2019,
arXiv:1904.07850. [Online]. Available: http://arxiv.org/abs/1904.07850

[6] J. Redmon and A. Farhadi,

improve-
ment,’’ 2018, arXiv:1804.02767. [Online]. Available: http://arxiv.org/abs/
1804.02767

‘‘YOLOv3: An incremental

[7] R. Girshick, ‘‘Fast R-CNN,’’ in Proc. IEEE Int. Conf. Comput. Vis.,

Dec. 2015, pp. 1440–1448.

[8] R. Huang, J. Pedoeem, and C. Chen, ‘‘YOLO-LITE: A real-time object
detection algorithm optimized for non-GPU computers,’’ in Proc. IEEE
Int. Conf. Big Data (Big Data), Dec. 2018, pp. 2503–2510.

[9] M. Gao, R. Yu, A. Li, V. I. Morariu, and L. S. Davis, ‘‘Dynamic zoom-
in network for fast object detection in large images,’’ in Proc. IEEE Conf.
Comput. Vis. Pattern Recognit., Jun. 2018, pp. 6926–6935.

[10] A. Pirinen and C. Sminchisescu, ‘‘Deep reinforcement learning of region
proposal networks for object detection,’’ in Proc. IEEE Conf. Comput. Vis.
Pattern Recognit., Jun. 2018, pp. 6945–6954.

[11] K. Rungsuptaweekoon, V. Visoottiviseth, and R. Takano, ‘‘Evaluating the
power efﬁciency of deep learning inference on embedded GPU systems,’’
in Proc. 2nd Int. Conf. Inf. Technol. (INCIT), Nov. 2017, pp. 1–5.

[12] H. Mao, S. Yao, T. Tang, B. Li, J. Yao, and Y. Wang, ‘‘Towards real-
time object detection on embedded systems,’’ IEEE Trans. Emerg. Topics
Comput., vol. 6, no. 3, pp. 417–431, Sep. 2018.

[13] J. Yu, K. Guo, Y. Hu, X. Ning, J. Qiu, H. Mao, S. Yao, T. Tang, B. Li,
Y. Wang, and H. Yang, ‘‘Real-time object detection towards high power
efﬁciency,’’ in Proc. Design, Autom. Test Eur. Conf. Exhib. (DATE),
Mar. 2018, pp. 704–708.

[14] F. N. Iandola, S. Han, M. W. Moskewicz, K. Ashraf, W. J. Dally, and
K. Keutzer, ‘‘SqueezeNet: AlexNet-level accuracy with 50x fewer param-
eters and < 0.5 MB model size,’’ 2016, arXiv:1602.07360. [Online]. Avail-
able: http://arxiv.org/abs/1602.07360

[15] A. Howard, M. Sandler, B. Chen, W. Wang, L.-C. Chen, M. Tan, G. Chu,
V. Vasudevan, Y. Zhu, R. Pang, H. Adam, and Q. Le, ‘‘Searching
for MobileNetV3,’’ in Proc. IEEE Int. Conf. Comput. Vis., Oct. 2019,
pp. 1314–1324.

[16] A. G. Howard, M. Zhu, B. Chen, D. Kalenichenko, W. Wang, T. Weyand,
M. Andreetto, and H. Adam, ‘‘MobileNets: Efﬁcient convolutional neu-
ral networks for mobile vision applications,’’ 2017, arXiv:1704.04861.
[Online]. Available: http://arxiv.org/abs/1704.04861

[17] M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L.-C. Chen,
‘‘MobileNetV2: Inverted residuals and linear bottlenecks,’’ in Proc. IEEE
Conf. Comput. Vis. Pattern Recognit., Jun. 2018, pp. 4510–4520.

[18] M. Tan, B. Chen, R. Pang, V. Vasudevan, and Q. V. Le, ‘‘MnasNet:
Platform-aware neural architecture search for mobile,’’ in Proc. IEEE/CVF
Conf. Comput. Vis. Pattern Recognit. (CVPR), Jun. 2018, pp. 2815–2823.
[19] X. Zhang, X. Zhou, M. Lin, and J. Sun, ‘‘ShufﬂeNet: An extremely
efﬁcient convolutional neural network for mobile devices,’’ in Proc. IEEE
Conf. Comput. Vis. Pattern Recognit., Jun. 2018, pp. 6848–6856.

[20] N. Ma, X. Zhang, H.-T. Zheng, and J. Sun, ‘‘ShufﬂeNet V2: Practical
guidelines for efﬁcient CNN architecture design,’’ in Proc. Eur. Conf.
Comput. Vis. (ECCV), 2018, pp. 116–131.

[21] I. Hubara, M. Courbariaux, D. Soudry, R. El-Yaniv, and Y. Bengio, ‘‘Bina-
rized neural networks,’’ in Proc. Adv. Neural Inf. Process. Syst., 2016,
pp. 4107–4115.

[22] I. Hubara, M. Courbariaux, D. Soudry, R. El-Yaniv, and Y. Bengio, ‘‘Quan-
tized neural networks: Training neural networks with low precision weights
and activations,’’ J. Mach. Learn. Res., vol. 18, no. 1, pp. 6869–6898, 2017.
[23] P. Bacchus, R. Stewart, and E. Komendantskaya, ‘‘Accuracy, training
time and hardware efﬁciency trade-offs for quantized neural networks on
FPGAs,’’ in Proc. Int. Symp. Appl. Reconﬁgurable Comput. Toledo, Spain:
Springer, 2020, pp. 121–135.

[24] C. Ding, S. Wang, N. Liu, K. Xu, Y. Wang, and Y. Liang, ‘‘REQ-YOLO:
A resource-aware, efﬁcient quantization framework for object detection
on FPGAs,’’ in Proc. ACM/SIGDA Int. Symp. Field-Programmable Gate
Arrays, Feb. 2019, pp. 33–42.

[25] Y. Kim, M. Imani, and T. S. Rosing, ‘‘Image recognition accelerator design
using in-memory processing,’’ IEEE Micro, vol. 39, no. 1, pp. 17–23,
Jan. 2019.

[26] J. Aloimonos, I. Weiss, and A. Bandyopadhyay, ‘‘Active vision,’’ Int.

J. Comput. Vis., vol. 1, no. 4, pp. 333–356, 1988.

[27] J. M. Findlay and I. D. Gilchrist, Active Vision: The Psychology of Looking

and Seeing, no. 37. London, U.K.: Oxford Univ. Press, 2003.

[28] H. Kolb, ‘‘Simple anatomy of the retina,’’ in Webvision: The Organization
of the Retina and Visual System. Salt Lake City, UT, USA: The Organiza-
tion of the Retina and Visual System, 1995, pp. 13–36.

[29] H. Strasburger, I. Rentschler, and M. Jüttner, ‘‘Peripheral vision and pattern

recognition: A review,’’ J. Vis., vol. 11, no. 5, p. 13, 2011.

[30] W. Kienzle, M. O. Franz, B. Schölkopf, and F. A. Wichmann, ‘‘Center-
surround patterns emerge as optimal predictors for human saccade targets,’’
J. Vis., vol. 9, no. 5, p. 7, May 2009.

[31] D. Purves, G. J. Augustine, D. Fitzpatrick, W. Hall, A.-S. LaMantia,
J. O. McNamara, and L. White, Neuroscience. Sunderland, MA, USA:
Sinauer Associates, 2001.

[32] A. Mack and I. Rock, Inattentional Blindness. Cambridge, MA, USA:

MIT Press, 1998.

[33] J. Li, F. Shi, W. Liu, D. Zou, Q. Wang, P.-K. J. Park, and H. E. Ryu,
‘‘Adaptive Temporal Pooling for Object Detection using Dynamic Vision
Sensor,’’ in Proc. Brit. Mach. Vis. Conf. (BMVC), T.-K. Kim, S. Zafeiriou,
G. Brostow, and K. Mikolajczyk, Eds. BMVA Press, Sep. 2017,
pp. 40.1–40.12. [Online]. Available: https://dx.doi.org/10.5244/C.31.40,
doi: 10.5244/C.31.40.

[34] N. F. Y. Chen, ‘‘Pseudo-labels for supervised learning on dynamic vision
sensor data, applied to object detection under ego-motion,’’ in Proc. IEEE
Conf. Comput. Vis. Pattern Recognit. Workshops, Jun. 2018, pp. 644–653.
[35] A. Dosovitskiy and J. Djolonga, ‘‘You only train once: Loss-conditional

training of deep networks,’’ in Proc. Int. Conf. Learn. Represent., 2020.

[36] E. Perez, F. Strub, H. D. Vries, V. Dumoulin, and A. Courville, ‘‘Film:
Visual reasoning with a general conditioning layer,’’ in Proc. 32nd AAAI
Conf. Artif. Intell., 2018, pp. 3942–3951.

[37] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, ‘‘Prox-
imal policy optimization algorithms,’’ 2017, arXiv:1707.06347. [Online].
Available: http://arxiv.org/abs/1707.06347

[38] T. P. Lillicrap, J. J. Hunt, A. Pritzel, N. Heess, T. Erez, Y. Tassa,
‘‘Continuous control with deep rein-
[Online]. Available:

learning,’’ 2015, arXiv:1509.02971.

D. Silver, and D. Wierstra,
forcement
http://arxiv.org/abs/1509.02971

[39] J. Schulman, S. Levine, P. Abbeel, M. Jordan, and P. Moritz, ‘‘Trust
region policy optimization,’’ in Proc. Int. Conf. Mach. Learn., 2015,
pp. 1889–1897.

[40] X. Zhu, Y. Xiong, J. Dai, L. Yuan, and Y. Wei, ‘‘Deep feature ﬂow for
video recognition,’’ in Proc. IEEE Conf. Comput. Vis. Pattern Recognit.,
Jul. 2017, pp. 2349–2358.

[41] X. Zhu, Y. Wang, J. Dai, L. Yuan, and Y. Wei, ‘‘Flow-guided feature
aggregation for video object detection,’’ in Proc. IEEE Int. Conf. Comput.
Vis., Oct. 2017, pp. 408–417.

[42] C. Hetang, H. Qin, S. Liu, and J. Yan, ‘‘Impression network for
video object detection,’’ 2017, arXiv:1712.05896. [Online]. Available:
http://arxiv.org/abs/1712.05896

[43] X. Zhu, J. Dai, X. Zhu, Y. Wei, and L. Yuan, ‘‘Towards high performance
video object detection for mobiles,’’ 2018, arXiv:1804.05830. [Online].
Available: http://arxiv.org/abs/1804.05830

[44] Z. Zhu, W. Wu, W. Zou, and J. Yan, ‘‘End-to-end ﬂow correlation tracking
with spatial-temporal attention,’’ in Proc. IEEE Conf. Comput. Vis. Pattern
Recognit., Jun. 2018, pp. 548–557.

[45] L. Zhang, Z. Lin, J. Zhang, H. Lu, and Y. He, ‘‘Fast video object segmen-
tation via dynamic targeting network,’’ in Proc. IEEE Int. Conf. Comput.
Vis., Oct. 2019, pp. 5582–5591.

[46] S. Jain, X. Wang, and J. E. Gonzalez, ‘‘Accel: A corrective fusion network
for efﬁcient semantic segmentation on video,’’ in Proc. IEEE Conf. Com-
put. Vis. Pattern Recognit. (CVPR), Jun. 2019, pp. 8866–8875.

[47] B. Mahasseni, S. Todorovic, and A. Fern, ‘‘Budget-aware deep semantic
video segmentation,’’ in Proc. IEEE Conf. Comput. Vis. Pattern Recognit.,
Jul. 2017, pp. 1029–1038.

[48] F. Xiao and Y. J. Lee, ‘‘Video object detection with an aligned spatial-
temporal memory,’’ in Proc. Eur. Conf. Comput. Vis. (ECCV), pp. 485–501,
2018.

124464

VOLUME 8, 2020

A. Rodriguez-Ramos et al.: Adaptive Inattentional Framework for Video Object Detection With Reward-Conditional Training

[49] M. Liu, M. Zhu, M. White, Y. Li, and D. Kalenichenko, ‘‘Looking
fast and slow: Memory-guided mobile video object detection,’’ 2019,
arXiv:1903.10172. [Online]. Available: http://arxiv.org/abs/1903.10172

[50] M. Zhu and M. Liu, ‘‘Mobile video object detection with temporally-
aware feature maps,’’ in Proc. IEEE Conf. Comput. Vis. Pattern Recognit.,
Jun. 2018, pp. 5686–5695.

[51] X. Chen, J. Yu, and Z. Wu, ‘‘Temporally identity-aware SSD with atten-
tional LSTM,’’ IEEE Trans. Cybern., vol. 50, no. 6, pp. 2674–2686,
Jun. 2020.

[52] S. S. Nabavi, M. Rochan, and W. Yang, ‘‘Future semantic segmentation
with convolutional LSTM,’’ 2018, arXiv:1807.07946. [Online]. Available:
http://arxiv.org/abs/1807.07946

[53] O. Russakovsky, J. Deng, H. Su, J. Krause, S. Satheesh, S. Ma, Z. Huang,
A. Karpathy, A. Khosla, M. Bernstein, A. C. Berg, and L. Fei-Fei, ‘‘Ima-
geNet large scale visual recognition challenge,’’ Int. J. Comput. Vis.,
vol. 115, no. 3, pp. 211–252, Dec. 2015.

[54] W. Han, P. Khorrami, T. Le Paine, P. Ramachandran, M. Babaeizadeh,
H. Shi, J. Li, S. Yan, and T. S. Huang, ‘‘Seq-NMS for video object
detection,’’ 2016, arXiv:1602.08465. [Online]. Available: http://arxiv.
org/abs/1602.08465

[55] K. Kang, H. Li, J. Yan, X. Zeng, B. Yang, T. Xiao, C. Zhang, Z. Wang,
R. Wang, X. Wang, and W. Ouyang, ‘‘T-CNN: Tubelets with convolutional
neural networks for object detection from videos,’’ IEEE Trans. Circuits
Syst. Video Technol., vol. 28, no. 10, pp. 2896–2907, Oct. 2018.

[56] K. Kang, W. Ouyang, H. Li, and X. Wang, ‘‘Object detection from video
tubelets with convolutional neural networks,’’ in Proc. IEEE Conf. Comput.
Vis. Pattern Recognit., Jun. 2016, pp. 817–825.

[57] H. Deng, Y. Hua, T. Song, Z. Zhang, Z. Xue, R. Ma, N. Robertson, and
H. Guan, ‘‘Object guided external memory network for video object detec-
tion,’’ in Proc. IEEE Int. Conf. Comput. Vis., Oct. 2019, pp. 6678–6687.

[58] L. Yang, Y. Fan, and N. Xu, ‘‘Video instance segmentation,’’ in Proc. IEEE

Int. Conf. Comput. Vis., Oct. 2019, pp. 5188–5197.

[59] H. Mao, T. Kong, and W. J. Dally, ‘‘CaTDet: Cascaded tracked detector for
efﬁcient object detection from video,’’ 2018, arXiv:1810.00434. [Online].
Available: http://arxiv.org/abs/1810.00434

[60] H. Luo, W. Xie, X. Wang, and W. Zeng, ‘‘Detect or track: Towards
cost-effective video object detection/tracking,’’ in Proc. AAAI Conf. Artif.
Intell., vol. 33, 2019, pp. 8803–8810.

[61] C. Feichtenhofer, A. Pinz, and A. Zisserman, ‘‘Detect to track and track to
detect,’’ in Proc. IEEE Int. Conf. Comput. Vis., Oct. 2017, pp. 3038–3046.
[62] Z. Jiang, P. Gao, C. Guo, Q. Zhang, S. Xiang, and C. Pan, ‘‘Video object
detection with locally-weighted deformable neighbors,’’ in Proc. AAAI
Conf. Artif. Intell., vol. 33, 2019, pp. 8529–8536.

[63] G. Bertasius, L. Torresani, and J. Shi, ‘‘Object detection in video with
spatiotemporal sampling networks,’’ in Proc. Eur. Conf. Comput. Vis.
(ECCV), 2018, pp. 331–346.

[64] K. Chen, J. Wang, S. Yang, X. Zhang, Y. Xiong, C. C. Loy, and D. Lin,
‘‘Optimizing video object detection via a scale-time lattice,’’ in Proc. IEEE
Conf. Comput. Vis. Pattern Recognit., Jun. 2018, pp. 7814–7823.

[65] Y. Li, J. Shi, and D. Lin, ‘‘Low-latency video semantic segmenta-
tion,’’ in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., Jun. 2018,
pp. 5997–6005.

[66] C. Ying and K. Fragkiadaki, ‘‘Depth-adaptive computational policies for
efﬁcient visual tracking,’’ in Proc. Int. Workshop Energy Minimization
Methods Comput. Vis. Pattern Recognit. Venice, Italy: Springer, 2017,
pp. 109–122.

[67] J. Supancic and D. Ramanan, ‘‘Tracking as online decision-making: Learn-
ing a policy from streaming videos with reinforcement learning,’’ in Proc.
IEEE Int. Conf. Comput. Vis., Oct. 2017, pp. 322–331.

[68] X. Hou and L. Zhang, ‘‘Saliency detection: A spectral residual approach,’’

in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., Jun. 2007, pp. 1–8.

[69] L. Itti, C. Koch, and E. Niebur, ‘‘A model of saliency-based visual attention
for rapid scene analysis,’’ IEEE Trans. Pattern Anal. Mach. Intell., vol. 20,
no. 11, pp. 1254–1259, 1998.

[70] M. Najibi, B. Singh, and L. Davis, ‘‘AutoFocus: Efﬁcient multi-scale infer-
ence,’’ in Proc. IEEE Int. Conf. Comput. Vis., Oct. 2019, pp. 9745–9755.
[71] C. Chen, M.-Y. Liu, O. Tuzel, and J. Xiao, ‘‘R-CNN for small object
detection,’’ in Proc. Asian Conf. Comput. Vis. Taipei, Taiwan: Springer,
2016, pp. 214–230.

[72] Y. Lu, T. Javidi, and S. Lazebnik, ‘‘Adaptive object detection using adja-
cency and zoom prediction,’’ in Proc. IEEE Conf. Comput. Vis. Pattern
Recognit., Jun. 2016, pp. 2351–2359.

[73] S. Gidaris and N. Komodakis, ‘‘Object detection via a multi-region and
semantic segmentation-aware CNN model,’’ in Proc. IEEE Int. Conf.
Comput. Vis., Dec. 2015, pp. 1134–1142.

[74] Y. Zhu, R. Urtasun, R. Salakhutdinov, and S. Fidler, ‘‘SegDeepM: Exploit-
ing segmentation and context in deep neural networks for object detec-
tion,’’ in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., Jun. 2015,
pp. 4703–4711.

[75] J. Ba, V. Mnih, and K. Kavukcuoglu, ‘‘Multiple object recognition
[Online]. Available:

with visual attention,’’ 2014, arXiv:1412.7755.
http://arxiv.org/abs/1412.7755

[76] S. Mathe, A. Pirinen, and C. Sminchisescu, ‘‘Reinforcement learning
for visual object detection,’’ in Proc. IEEE Conf. Comput. Vis. Pattern
Recognit., Jun. 2016, pp. 2894–2902.

[77] J. C. Caicedo and S. Lazebnik, ‘‘Active object localization with deep
reinforcement learning,’’ in Proc. IEEE Int. Conf. Comput. Vis., Dec. 2015,
pp. 2488–2496.

[78] B. Uzkent and S. Ermon, ‘‘Learning when and where to zoom with deep
reinforcement learning,’’ 2020, arXiv:2003.00425. [Online]. Available:
http://arxiv.org/abs/2003.00425

[79] B. Uzkent, C. Yeh, and S. Ermon, ‘‘Efﬁcient object detection in large
images using deep reinforcement learning,’’ in Proc. IEEE Winter Conf.
Appl. Comput. Vis., Mar. 2020, pp. 1824–1833.

[80] Z. Jie, X. Liang, J. Feng, X. Jin, W. Lu, and S. Yan, ‘‘Tree-structured
reinforcement learning for sequential object localization,’’ in Proc. Adv.
Neural Inf. Process. Syst., 2016, pp. 127–135.

[81] W. Liu, D. Anguelov, D. Erhan, C. Szegedy, S. Reed, C.-Y. Fu, and
A. C. Berg, ‘‘SSD: Single shot multibox detector,’’ in Proc. Eur. Conf.
Comput. Vis. Amsterdam, The Netherlands: Springer, 2016, pp. 21–37.

[82] K. He, X. Zhang, S. Ren, and J. Sun, ‘‘Deep residual learning for
image recognition,’’ in Proc. IEEE Conf. Comput. Vis. Pattern Recognit.,
Jun. 2016, pp. 770–778.

[83] A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen,
Z. Lin, N. Gimelshein L. Antiga, and A. Desmaison, ‘‘PyTorch: An imper-
ative style, high-performance deep learning library,’’ in Proc. Adv. Neural
Inf. Process. Syst., 2019, pp. 8024–8035.

[84] M. Abadi et al., ‘‘TensorFlow: A system for large-scale machine learning,’’
in Proc. 12th USENIX Symp. Oper. Syst. Design Implement. (OSDI), 2016,
pp. 265–283.

ALEJANDRO RODRIGUEZ-RAMOS (Graduate
Student Member, IEEE) received the M.Sc. degree
in telecommunication engineering (major in elec-
tronics and micro-electronics) from the Univer-
sidad Politécnica de Madrid (UPM), Madrid,
Spain, in 2015, where he is currently pursuing
the Ph.D. degree with the Centre for Automa-
tion and Robotics, Computer Vision and Aerial
Robotics (CVAR) Group. H was a Visiting
Researcher with the Aerospace Controls Labora-
tory (ACL), Massachusetts Institute of Technology (MIT), from October to
December 2018, for three months. He worked for more than a year in the
aerospace sector, contributing to projects from the European Space Agency
(ESA). He is currently a Researcher with the Computer Vision and Aerial
Robotics (CVAR) Group, Centre for Automation and Robotics, Universidad
Politécnica de Madrid. His research interests include deep reinforcement
learning techniques applied to aerial robotics, deep learning, aerial robotics,
and image processing. He has received several international prizes in UAV
competitions, such as IMAV2016, IMAV 2017, and MBZIRC 2020 (Team
Leader).

JAVIER RODRIGUEZ-VAZQUEZ received the
B.Sc. degree in computer engineering (double
major in hardware engineering and computer sci-
ence) and the M.Sc. degree in systems engineer-
ing and computing research from the Universidad
de Cádiz (UCA), Cádiz, Spain, in July 2015 and
March 2017, respectively. He is currently pursuing
the Ph.D. degree in artiﬁcial intelligence with the
Universidad Politécnica de Madrid (UPM). His
research interest includes deep learning methods
for solving computer vision tasks, with a special interest in object detec-
tion and image segmentation. He has received an international prize in the
MBZIRC 2020 Competition.

VOLUME 8, 2020

124465

A. Rodriguez-Ramos et al.: Adaptive Inattentional Framework for Video Object Detection With Reward-Conditional Training

CARLOS SAMPEDRO (Member, IEEE) received
the B.Sc. degree in industrial engineering (major
in industrial electronics), obtaining the best marks
degree award, and the master’s degree in automa-
tion and robotics from the Universidad Politécnica
de Madrid (UPM), Madrid, Spain, in July 2011
and 2014, respectively, and the Ph.D. degree (cum
laude) in automation and robotics, in December
2019. He was a Visiting Researcher with Ari-
zona State University, AZ, USA, from Septem-
ber 2015 to December 2015. He is actively involved in the development
of deep reinforcement learning algorithms for autonomous navigation and
control of unmanned aerial vehicles (UAVs). His research interest includes
the applications of learning-based techniques for solving computer vision
problems, with a special interest in object detection and recognition using
deep learning techniques. He received the Predoctoral Grant from the Uni-
versidad Politécnica de Madrid, in January 2017.

PASCUAL CAMPOY (Senior Member, IEEE) has
been a Visiting Professor with Tongji University,
Shanghai, China, and QUT, Australia. He is cur-
rently a Lecturer in control, machine learning,
and computer vision. He is also a Full Professor
in automation and robotics with the Universidad
Politécnica de Madrid (UPM), Madrid, Spain, and
a Visiting Professor with TU Delft, The Nether-
lands. He is leading the Centre for Automation
and Robotics (CAR), Computer Vision and Aerial
Robotics (CVAR) Research Group. He has been the Head Director of
over 40 research and development projects, including research and devel-
opment European projects, national research and development projects, and
over 25 technological transfer projects directly contracted with the industry.
He is the author of over 200 international scientiﬁc publications. He holds
nine patents, three of them registered internationally. He has received several
international prizes in unmanned aerial vehicle (UAV) competitions, such
as IMAV 2012, IMAV 2013, IARC 2014, IMAV2016, IMAV 2017, and
MBZIRC 2020.

124466

VOLUME 8, 2020

